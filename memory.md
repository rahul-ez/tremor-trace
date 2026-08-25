# Memory - Tremor Signal Processing Build Handoff

Last updated: 2026-08-20

## What was built

### Acquisition logger and calibration

- Updated `scripts/run_acquisition_check.py` so recordings append by default instead of truncating an existing CSV.
- Added explicit `--overwrite` support for intentional replacement of a recording.
- Improved serial auto-detection to select a unique USB serial device. The ESP32/CH340 device was observed on COM9.
- Declared `pyserial` in `pyproject.toml`.
- Added logger regression tests in `tests/test_acquisition_logger.py`.
- Corrected Feature 11 in `signal_processing/calibration.py`:
	- Gyroscope offsets remain stationary means.
	- Accelerometer X/Y offsets use stationary means.
	- The Z offset subtracts expected +1g sensitivity so calibration preserves gravity.
	- Calibration requires the device to be flat with Z vertical.
	- Input shape is strictly `(n_samples, 6)`.
	- Unused hardcoded conversion constants were removed.
- Updated `tests/test_calibration.py` for preserved +1g Z calibration and invalid 1D, 3D, and wrong-column shapes.

### Features 13-15: filtering and strongest axis

- Added `signal_processing/filtering.py`.
- Implemented `remove_baseline()`:
	- Accepts 1D or 2D signals.
	- Subtracts the mean from a 1D signal or independently from each 2D axis.
	- Preserves shape, dtype convention, and physical units.
	- This is separate from sensor calibration and removes the calibrated gravity/DC baseline.
- Implemented `bandpass_filter()`:
	- Uses `scipy.signal.butter` and `scipy.signal.filtfilt`.
	- Supports 1D and 2D signals, filtering along the sample axis.
	- Is explicitly documented as offline and non-causal.
	- Validates sample rate, band limits, order, and signal shape.
- Completed `select_strongest_axis()` in `signal_processing/axis_handling.py`:
	- Filters each of three axes in the tremor band.
	- Scores each axis using mean-square filtered power.
	- Returns the original calibrated axis with the greatest tremor-band power, not the filtered intermediate signal.
	- Uses configured sample rate, tremor band, and filter order by default, with optional explicit arguments for testing.
- Replaced the Feature 15 stub tests in `tests/test_axis_handling.py` with synthetic 6 Hz selection tests.
- Added `tests/test_filtering.py` for baseline removal, pass-band behavior, attenuation, invalid shapes, and invalid bands.

### Features 16-18: windows and spectral analysis

- Added `signal_processing/windowing.py`.
- Implemented `segment_windows()`:
	- Accepts 1D or 2D signals and preserves channel structure.
	- Uses complete fixed-size windows only.
	- Discards incomplete trailing samples rather than padding.
	- Uses `n_windows = (n_samples - window_samples) // step_samples + 1`.
	- Returns an empty correctly shaped array when the signal is shorter than one window.
- Added `signal_processing/spectral_analysis.py`.
- Implemented `compute_welch_psd()` using `scipy.signal.welch`.
- Implemented `tremor_band_power()` using numerical trapezoidal integration over selected frequency bins.
- Implemented `total_power()` using numerical trapezoidal integration over the supplied PSD range.
- Implemented `power_ratio()` with the required `+1e-12` denominator guard.
- Implemented `dominant_frequency()` as the maximum PSD bin within the tremor band.
- Added `tests/test_windowing.py` and `tests/test_spectral_analysis.py` covering geometry, overlap, trailing discard, 6 Hz detection, numerical integration, zero-power handling, and band-limited peak selection.

### Features 19-21 numeric functionality

- Added `signal_processing/time_domain.py`.
- Implemented `compute_rms()` as `sqrt(mean(x^2))` over a 1D filtered tremor window.
- Implemented `compute_variance()` over a 1D filtered tremor window.
- Both preserve physical units conceptually and do not normalize RMS.
- Added `spectral_entropy()` to `signal_processing/spectral_analysis.py`:
	- Uses `p_i = PSD_i / sum(PSD)`.
	- Computes `-sum(p_i * log2(p_i))` over nonzero probabilities.
	- Normalizes by `log2(N)`.
	- Returns 0 for an all-zero PSD or a one-bin PSD.
	- Clips numerical roundoff into `[0, 1]`.
- Added `signal_processing/feature_extraction.py`.
- Implemented `extract_features()` as the feature assembly boundary:
	- Receives a 1D filtered tremor analysis window for spectral/time-domain features.
	- Receives separate `(n_samples, 3)` calibrated accelerometer and gyroscope windows for magnitude features.
	- Uses existing owner modules rather than duplicating signal-processing logic.
	- Returns the shared `tremor_system.types.FeatureVector`.
	- Metadata fields are `subject_id`, `session_id`, and `window_id` only.
	- `FeatureVector.to_dict()` exposes exactly nine numeric ML features in documented order:
		`tremor_band_power`, `total_power`, `power_ratio`, `dominant_frequency_hz`, `rms_amplitude`, `variance`, `spectral_entropy`, `accel_magnitude`, `gyro_magnitude`.
	- Accelerometer magnitude is the mean per-sample Euclidean norm in g.
	- Gyroscope magnitude is the mean per-sample Euclidean norm in deg/s.
- Added `tests/test_time_domain.py`, `tests/test_feature_extraction.py`, and entropy/magnitude/schema coverage in `tests/test_spectral_analysis.py`.

## Decisions made

- The project remains simulation-first. No physical electrical stimulation path is added.
- Raw recordings are append-only by default; replacement requires `--overwrite`.
- Feature 11 preserves the stationary +1g Z gravity component. Baseline/gravity removal is Feature 13 and is not folded into calibration.
- v1 Butterworth filter order is `4`. `config/system_config.yaml` documents alternatives `2` and `6`; all code remains configuration-driven.
- Baseline removal is per-axis mean subtraction rather than a high-pass filter.
- Band-pass filtering is offline/non-causal and uses zero-phase `filtfilt`.
- Strongest-axis selection ranks axes by mean-square power after tremor-band filtering, but returns the original calibrated axis.
- v1 sensor sampling rate is `100.0 Hz`.
- v1 window length is `2.0 s`, equal to `200` samples at 100 Hz.
- v1 overlap is `50%`, equal to a `100`-sample step for 200-sample windows.
- Incomplete trailing windows are discarded, never zero-padded.
- Welch PSD is mandatory; raw FFT is not the standard spectral path.
- v1 Welch `nperseg` is `200`.
- Spectral power is integrated numerically over frequency bins, not summed without bin spacing.
- Tremor band defaults to `[4.0, 12.0] Hz` from configuration.
- Dominant frequency is the maximum PSD bin within the configured tremor band.
- RMS is unnormalized and retains the input signal's physical units.
- Spectral entropy is normalized Shannon entropy over the PSD bins and safely ignores zero bins.
- Accelerometer and gyroscope magnitude features are window-level mean Euclidean magnitudes and are separate from the signal representation used for spectral analysis.
- Numeric ML features exclude subject/session/window metadata.
- The visual raw-versus-filtered/PSD baseline script is a separate integration task and was intentionally not implemented as part of the numeric Features 19-21 work.

## Problems solved

- The acquisition logger no longer erases existing recordings.
- Serial auto-detection no longer falls back to stdin when the ESP32/CH340 port is available alongside Bluetooth ports.
- Calibration no longer removes the physical +1g gravity component from the stationary Z axis.
- Calibration malformed-shape failures are now explicit and early.
- The former strongest-axis placeholder is now functional and tested.
- The configuration tests were updated when `filter_order` changed from unresolved to v1 value `4`, and when the existing `axis_strategy` default was correctly expected as `per_axis`.
- The configuration tests were updated when `window_length_s` changed from unresolved to v1 value `2.0`.
- The progress tracker was corrected so it does not falsely claim the separate visual-check script is complete.

## Current state

- The recorded `data/raw/subj01/sess01/raw_stream.csv` contains 10,074 valid samples over 100.73 seconds at exact 100 Hz.
- COM9 was verified live and produced valid samples at approximately 100 Hz.
- The full pytest suite passes: 70 tests passed on the latest run.
- Diagnostics report no errors for the new time-domain, spectral-analysis, feature-extraction, and test files.
- Phase 3 numeric signal-processing work through feature assembly is implemented and tested.
- `context/progress-tracker.md` currently records Features 10-20 as complete, with the original Feature 21 visual-check script still unchecked. The tracker text notes that magnitude features are included in feature assembly because the clarified request used Feature 21 for numeric magnitude functionality.
- The last unrelated acquisition-check command exited with code 1 when run without an available usable serial invocation; this was not changed during the signal-processing work. The latest `pytest -q` exited successfully.
- No ML, estimation, controller, simulation, validation, or visual-check script implementation has been started in this work.

## Next session starts with

- Start with the still-pending Phase 3 integration task: implement `scripts/run_pipeline_baseline.py` only if the developer wants the visual raw-versus-filtered/PSD check next.
- Before editing, read `AGENTS.md` and the six context files in their required order if the new session does not already have that context.
- If the developer instead wants to proceed directly to Phase 4, begin Feature 22, the subject-level dataset builder, after confirming whether the visual-check gate is intentionally being deferred.
- Preserve the current configuration: sample rate 100 Hz, filter order 4, window length 2.0 s, overlap 50%, tremor band 4-12 Hz, and Welch `nperseg=200`.
- Run focused tests immediately after the first edit, then the full pytest suite before updating `context/progress-tracker.md`.

## Open questions

- Whether to implement the separate Feature 21 visual-check script before Phase 4 ML work, or explicitly defer it.
- The project context still contains several later-phase TBD values, including ML thresholds, controller thresholds/bounds, simulation timestep/latency, and final experimental choices. Do not silently replace those with hardcoded production values.
- The feature extraction API currently separates `analysis_window`, `accel_window`, and `gyro_window` to preserve architecture boundaries. Any future caller integration should retain that separation unless the developer explicitly changes the contract.

## Session Update - Feature 21 Complete

### What was built

- Added `scripts/run_pipeline_baseline.py` for the Phase 3 visual verification pipeline.
- The script loads a recorded raw CSV, runs calibration, accelerometer axis handling, baseline removal, tremor-band filtering, overlapping windowing, and Welch PSD analysis using the existing production modules.
- The script saves a two-panel PNG containing calibrated acceleration/filtered tremor signals and a Welch PSD with the configured 4-12 Hz band and dominant-frequency marker.
- Added `tests/test_pipeline_baseline.py` to verify that the integration pipeline creates a non-empty plot artifact.
- Generated `data/processed/subj01/sess01/pipeline_baseline.png` from the recorded session.

### Verification

- Direct execution succeeded with `python scripts/run_pipeline_baseline.py`.
- The recorded session's first complete window produced a dominant tremor-band frequency of `9.00 Hz`.
- Full pytest suite passes: `71 passed`.
- Diagnostics report no errors in the new script or integration test.
- Updated `context/progress-tracker.md`: Feature 21 is complete and the project advances to Phase 4, Feature 22 (subject-level dataset builder).

### Current state

- Phase 3 signal processing is complete, including the visual raw-versus-filtered/PSD verification gate.
- The next implementation target is Phase 4 Feature 22: dataset construction with subject-level train/validation/test splitting.

## Session Update - Features 22-27 Complete (Phase 4)
 
### What was built
 
- Added `ml/dataset_builder.py`: `build_feature_table`, `load_dataset`, session-level `assign_labels`/`default_session_label`, and `get_subject_level_folds` (GroupKFold).
- Added `ml/baseline_detector.py`: non-ML `power_ratio` threshold detector and `evaluate_baseline`.
- Added `ml/train.py`: `train_and_scale` (LogisticRegression/SVM/RandomForest) and `save_artifacts`.
- Added `ml/evaluate.py`: `evaluate_model` (precision/recall/F1/sensitivity/FPR), `evaluate_candidates_across_folds`, `aggregate_fold_results`.
- Added `ml/model_selection.py`: `select_model`, smallest artifact meeting `min_recall`/`min_precision`.
- Added `ml/inference.py`: sole `predict()` entry point returning the documented `InferenceResult` contract.
- Added `scripts/build_features.py`: batch raw_stream.csv -> per-session and consolidated `features.csv`.
- Added `scripts/run_training.py`: orchestrates dataset_builder -> baseline -> train -> evaluate -> model_selection -> save.
- Added matching test files for all six `ml/` modules.
### Decisions made
 
- Session-label convention confirmed by the developer: `session_id` ending `01` = stationary "hand resting in glove" (label False, no tremor); ending `02` = deliberate tremor-like motion (label True). Implemented as `ml/dataset_builder.py::default_session_label`.
- `sess01` (confirmed stationary) is also the correct source for per-subject sensor calibration. `scripts/build_features.py::estimate_subject_offsets` estimates one offset vector per subject from their `sess01` recording and reuses it for all of that subject's sessions, rather than estimating offsets per-session. Falls back to zero-offset calibration only when a subject has no valid stationary session (`subj03`'s `sess01` fails z-axis offset validation even though it is a rest recording, likely device-flatness during that specific recording; flagged via a WARNING log, not silently absorbed).
- `config.signal.axis_strategy = "per_axis"` has no single-channel meaning for a scalar-per-window feature vector; `scripts/build_features.py::_select_analysis_signal` falls back to strongest-axis selection in that case, matching the precedent already set by `scripts/run_pipeline_baseline.py`. Logged at INFO level. This is a pipeline-integration decision, not a resolution of architecture.md Open Question #3 (final axis-handling strategy after comparative evaluation), which remains open.
- SVM candidate uses `CalibratedClassifierCV(SVC(...), method="sigmoid", ensemble=False)` instead of `SVC(probability=True)`, since the latter is deprecated as of scikit-learn 1.9 and scheduled for removal in 1.11.
- Final model artifacts are refit on the full dataset after cross-validated evaluation; the fold-level models exist only to produce honest held-out metrics and are not saved.
### Verification
 
- Full pytest suite passes: 87 passed, 8 skipped (skips require real recorded data/trained models on disk and are exercised once `scripts/build_features.py` + `scripts/run_training.py` have been run against real data).
- Ran end-to-end against the developer's real recordings (8 subjects, 16 sessions, 1616 windows): `logistic_regression` selected (precision 1.0, recall ~0.95, 943-byte artifact); `random_forest` precision ~0.996/recall ~0.994; `svm` precision ~0.93/recall ~0.97. All three candidates clear `min_recall=0.7`/`min_precision=0.7`.
- `tests/test_inference.py::test_predict_on_recorded_deliberate_tremor_session` (previously skipped) passes against the real trained model, confirming `label=True` on an actual tremor session.
### Current state
 
- Phase 4 (Features 22-27) is complete and verified against real recorded data on the developer's machine (Windows, scikit-learn 1.8+).
- `data/models/` contains `model_logistic_regression_v1.pkl`, `model_svm_v1.pkl`, `model_random_forest_v1.pkl`, `scaler_v1.pkl`, `evaluation_results.csv`, `evaluation_results_by_fold.csv`.
- The next implementation target is Phase 5, Feature 28: dominant frequency estimation module.
### Open questions
 
- `subj03` has no usable stationary calibration recording (both sessions fall back to zero-offset calibration); consider re-recording `subj03/sess01` or accepting the current fallback.
- architecture.md Open Questions #3 (final axis-handling strategy) and #4 (final ML model selection) still list `axis_strategy` and model choice as open pending comparative evaluation / ESP32 resource confirmation; this session's choices (strongest-axis fallback, logistic_regression selection) are v1/pipeline decisions, not a closure of those architecture-level questions. Feature 32 (ESP32 resource-fit report) is still pending real on-device measurement.

## Session Update - Features 28-31 Complete (Phase 5)

### What was built

- Added `estimation/frequency_estimation.py`: `estimate_frequency(freqs_hz, psd, tremor_band_hz, min_peak_to_mean_ratio)`, wrapping `signal_processing/spectral_analysis.py::dominant_frequency()` with a peak-vs-mean-power guard that returns `None` for a flat/noise-only spectrum instead of an arbitrary bin.
- Added `estimation/amplitude_estimation.py`: `estimate_amplitude(filtered_signal)`, thin wrapper around `compute_rms()`.
- Added `estimation/phase_estimation.py`: `estimate_phase(filtered_signal)`, Hilbert-transform instantaneous phase at the window's final sample; returns `None` for a constant (non-oscillating) signal.
- Added `config.estimation.phase_enabled: bool` (default `false`) to `system_config.yaml`/`tremor_system/config.py`.
- Updated `ml/inference.py::predict()`: signature now takes `analysis_window`, `sample_rate_hz`, and an optional `config` (defaults to `load_config()`). `dominant_frequency_hz`/`amplitude` are now computed live from the window via Features 28-29 instead of copied from the input `features` dict. `phase` is computed via `estimate_phase()` only when `config.estimation.phase_enabled` is `True`; otherwise stays `None`.
- Added matching test files for all three `estimation/` modules, and extended `tests/test_inference.py` for the new `predict()` signature plus explicit phase-gating tests (disabled by default; populated when the config flag is set).

### Decisions made

- `min_peak_to_mean_ratio` (default 1.5) in `estimate_frequency()` is an experimental heuristic threshold -- architecture.md's Open Questions has no assigned value for this; not yet validated against labeled tremor/no-tremor spectra beyond the synthetic sine test.
- `estimate_phase()` returns phase at the window's *final* sample (matches the "phase right now" use case a real-time controller would need), not a windowed/averaged phase. Hilbert transform boundary effects mean this is less accurate near the very edge of a short window -- not yet quantified against a margin (Feature 31's verification explicitly calls this margin TBD).
- `predict()` takes `config` as an optional parameter (not always calling `load_config()` internally) so a real-time inference loop can load config once and reuse it across many calls rather than re-reading the YAML every window.
- Initial phase-estimation test had a bug (not in the estimation code): compared a sine wave's own argument against Hilbert's phase output, which uses a cosine-phase convention offset by exactly pi/2 from that. Fixed the test to use `cos()`, not `estimate_phase()`.
- Feature 31 was initially built and tested standalone but not actually wired into `predict()` (config flag existed but nothing read it) -- caught and fixed in the same session before considering Phase 5 closed.

### Verification

- Full pytest suite passes: 99 passed, 8 skipped (same 8 skips as Phase 4, gated on real recorded data/trained models being present locally).
- Ran end-to-end against the developer's real recordings: on a genuine `subj02/sess02` tremor window, `predict()` returned `label=True`, `dominant_frequency_hz=4.5` (inside the 4-12 Hz tremor band), `amplitude=0.217` (physically plausible for a hand tremor), `phase=None` (default-disabled, as expected).
- Confirmed via `dataclasses.replace()` on a loaded `Config` that flipping `estimation.phase_enabled=True` does populate a real phase value in `[-pi, pi]`.

### Current state

- Phase 5 (Features 28-31) is complete, verified against real recorded data, and phase estimation is fully wired (not just built standalone).
- `ml/inference.py::predict()`'s signature changed from Feature 27's version (`analysis_window`/`sample_rate_hz` are now required positional args); no other module in this repo calls `predict()` yet, so this had no other call sites to update.
- The next implementation target is Phase 6, Feature 32: synthetic tremor signal generator.

### Open questions

- `min_peak_to_mean_ratio` and the phase-accuracy margin (Feature 31's verification) are both unvalidated heuristics -- revisit once more labeled recordings exist.
- Everything under Open questions in the Phase 4 session update above is still open (subj03 calibration fallback, architecture.md Open Questions #3/#4 not formally closed).