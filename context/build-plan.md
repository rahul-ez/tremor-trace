# Build Plan

## Core Principle

Build the system as a sequence of independently testable vertical slices, in strict dependency order, verifying each feature against recorded data, synthetic signals, simulation output, or hardware behavior before starting the next. No phase advances until every feature in it has passed its stated verification — later phases (ML, controller, closed-loop integration) must never mask a broken earlier stage (acquisition, filtering, PSD).

---

## Phase 1 — Foundation

Establish project structure, the single configuration source of truth, shared data-contract types, and test infrastructure. No tremor detection or control logic is implemented in this phase.

### 01 Repository and Folder Structure

**Goal:** Create the full directory skeleton defined in `architecture.md` so every later feature has a known location to land in.

**Implementation:**
- Create top-level folders: `firmware/`, `config/`, `data/{raw,processed,features,models,simulation,validation}/`, `signal_processing/`, `ml/`, `estimation/`, `controller/`, `simulation/`, `validation/experiments/`, `visualization/`, `tests/`, `scripts/`.
- Add empty `__init__.py` files to every Python package directory.
- Add a `.gitignore` excluding `data/raw/`, `data/processed/`, `data/models/*.pkl` (large/generated artifacts) but tracking folder structure via `.gitkeep`.

**Inputs / Outputs:**
- Output: full folder tree matching `architecture.md` → Folder Structure, committed to version control.

**Verification:**
- Run a directory-listing check (`find . -type d`) and diff against the folder structure documented in `architecture.md`; every listed folder must exist.

---

### 02 Central Configuration File and Loader

**Goal:** Establish `config/system_config.yaml` as the single source of truth for every tunable parameter, with a typed Python loader.

**Implementation:**
- Create `config/system_config.yaml` with sections: `sensor` (sample_rate_hz, i2c_clock_hz, accel_range_g, gyro_range_dps), `signal` (tremor_band_hz, voluntary_band_hz, filter_order, window_length_s, window_overlap_pct, axis_strategy), `ml` (confidence_threshold, random_seed), `controller` (severity_threshold, target_suppression_pct, hysteresis_pct, max_delta_per_step, param_bounds), `simulation` (timestep_s, latency_ms).
- Mark every value not yet experimentally determined explicitly as `# TBD` with a comment referencing the corresponding `architecture.md` Open Question number.
- Implement `tremor_system/config.py` with `load_config() -> Config`, using a typed dataclass (or Pydantic-free simple dataclass per `code-standards.md`) that fails fast (raises) on a missing required key.

**Inputs / Outputs:**
- Input: `config/system_config.yaml`.
- Output: `Config` object consumed by every later Python module.

**Verification:**
- `pytest tests/test_config.py`: asserts `load_config()` returns a populated object, all required keys are present, and TBD values are exposed as `None` or an explicit `Tbd` sentinel rather than a silently guessed number.

---

### 03 Shared Data-Contract Types

**Goal:** Define the shared dataclasses/schemas (`StimParams`, `SimResult`, feature-vector schema, ML inference output) referenced throughout `architecture.md` so every module imports the same types instead of redefining them.

**Implementation:**
- Create a shared module (location TBD — architecture does not name an exact file; place under e.g. `tremor_system/types.py` and treat exact placement as an implementation detail to confirm against later usage) containing: `StimParams`, `SimResult`, `InferenceResult`/ML output dataclass, `WindowResult`.
- Field names, types, and units must exactly match `architecture.md` → Interfaces & Data Contracts and `code-standards.md` → Data Contracts.

**Inputs / Outputs:**
- Output: importable dataclasses with no business logic, used by `ml/`, `controller/`, `simulation/`.

**Verification:**
- `pytest tests/test_types.py`: instantiate each dataclass with valid sample values and assert field names/types match the documented contract exactly (field-by-field comparison against a hardcoded expected schema in the test).

---

### 04 Test and Verification Infrastructure

**Goal:** Stand up `pytest` as the test runner and establish the convention that every later feature ships with a corresponding test file.

**Implementation:**
- Add `pytest` to Python dependencies (per `library-docs.md` → Approved Dependencies).
- Create `tests/` package mirroring the module structure (`tests/test_config.py`, `tests/test_types.py`, placeholders for `tests/test_filtering.py`, `tests/test_windowing.py`, etc. to be filled in by later features).
- Configure `pytest.ini` or `pyproject.toml` `[tool.pytest.ini_options]` with `testpaths = ["tests"]`.

**Inputs / Outputs:**
- Output: `pytest` runs successfully with zero collected failures on the (currently minimal) test suite.

**Verification:**
- Run `pytest -v` from the project root; all currently-defined tests (Features 02–03) pass, and the run exits with status 0.

---

## Phase 2 — MPU6050 Acquisition

Implement the ESP32 + MPU6050 acquisition layer and establish a reliable, verified raw-data pipeline before any processing logic depends on it.

### 05 MPU6050 I²C Driver

**Goal:** Implement `firmware/src/mpu6050_driver.cpp/.h` so the ESP32 can initialize and configure the MPU6050 over I²C.

**Implementation:**
- Implement `setupI2C()`, `writeRegister()`, `i2cReadBlock()` per the patterns in `library-docs.md` → Wire (I²C).
- Implement `setupMPU6050()`: configure DLPF (`0x03`), accel range (±2g), gyro range (±250°/s), per `architecture.md` → Hardware Architecture.
- Pins/clock: SDA=GPIO21, SCL=GPIO22, 400 kHz, as constants at the top of the file.

**Inputs / Outputs:**
- Input: I²C bus at the configured pins/clock.
- Output: MPU6050 configured and readable; `writeRegister`/`i2cReadBlock` return success/failure status.

**Verification (hardware-dependent):**
- Flash to a physical ESP32 with MPU6050 attached. Read the `WHO_AM_I` register and confirm it returns the expected MPU6050 device ID over serial debug output. Confirm `writeRegister`/`i2cReadBlock` return `true` on a stationary, connected device.

---

### 06 Timed 100 Hz Acquisition Loop

**Goal:** Acquire raw accel/gyro samples at a stable ~100 Hz using a hardware timer, per `library-docs.md` → Arduino/ESP32 Arduino Core.

**Implementation:**
- Implement `firmware/src/sampling_timer.cpp/.h`: `setupSamplingTimer()` and the `onSampleTimer()` ISR/flag pattern.
- Implement `readMPU6050Raw(RawSample&)` in `mpu6050_driver.cpp`, reading the full 14-byte block and populating `timestamp_us` via `micros()`.
- Wire the flag-driven read into `firmware/src/main.cpp`'s `loop()`.

**Inputs / Outputs:**
- Output: a `RawSample{timestamp_us, ax, ay, az, gx, gy, gz}` produced approximately every 10 ms.

**Verification (hardware-dependent):**
- Log 10,000 consecutive samples' `timestamp_us` values over serial. Compute inter-sample interval statistics (mean, std dev) in a small Python script; mean interval must be within an experimentally acceptable tolerance of 10,000 µs (exact tolerance TBD — record actual measured jitter as the baseline).

---

### 07 Serial CSV Output Protocol

**Goal:** Stream each acquired sample over USB serial in the exact documented CSV format.

**Implementation:**
- Implement `firmware/src/serial_protocol.cpp/.h`: `writeSampleCsv(const RawSample&)` producing `timestamp_us,ax,ay,az,gx,gy,gz\n` at 115200 baud, using `snprintf` into a fixed buffer (no `String` concatenation), per `code-standards.md` → ESP32/C++ Firmware.
- Call `writeSampleCsv()` from `main.cpp`'s `loop()` after a successful read.

**Inputs / Outputs:**
- Input: `RawSample`.
- Output: one CSV line per sample on the serial port.

**Verification (hardware-dependent):**
- Capture 60 seconds of serial output to a file using a PC-side serial logger. Parse every line with a simple CSV parser and confirm: (a) every line has exactly 7 comma-separated fields, (b) `timestamp_us` is monotonically increasing, (c) zero malformed lines in a stationary-device capture.

---

### 08 I²C Failure Handling

**Goal:** Ensure the acquisition loop degrades safely (skips, logs, continues) rather than transmitting corrupted data on I²C failure.

**Implementation:**
- In `main.cpp`'s `loop()`, check `readMPU6050Raw()`'s return value; on `false`, emit `ERR,I2C_READ_FAILED` and skip the CSV write for that cycle, per `code-standards.md` → Error Handling.

**Inputs / Outputs:**
- Output: distinct `ERR,<REASON>` marker lines interleaved with normal CSV data lines on I²C failure.

**Verification (hardware-dependent):**
- Deliberately interrupt the I²C connection briefly (e.g., loose wire) during a capture session and confirm the serial log contains `ERR,I2C_READ_FAILED` lines instead of malformed/garbage CSV rows, and that normal CSV output resumes once the connection is restored.

---

### 09 PC-Side Raw Data Logger

**Goal:** Provide a PC-side script that reads the live serial stream and writes it to `data/raw/<subject_id>/<session_id>/raw_stream.csv`.

**Implementation:**
- Create `scripts/run_acquisition_check.py` (this script also satisfies the Milestone-1 verification goal from `architecture.md` → Signal Processing Architecture): opens the serial port at 115200 baud, reads lines, discards malformed lines and `ERR,*` marker lines from the data file (optionally logging them separately), writes valid samples to the target CSV path.
- Accept `--subject-id` and `--session-id` as command-line arguments to construct the output path.

**Inputs / Outputs:**
- Input: live serial stream from Feature 07/08.
- Output: `data/raw/<subject_id>/<session_id>/raw_stream.csv` matching the documented raw-sample schema.

**Verification (hardware-dependent):**
- Run the script for 30 seconds with the ESP32 connected and stationary. Confirm the output CSV exists, has the correct header/columns, a plausible row count (~30 × 100 = ~3000 rows, allowing for measured jitter), and no malformed rows.

---

## Phase 3 — Signal Processing

Implement the signal-processing pipeline in strict dependency order: calibration → axis handling → baseline/gravity removal → filtering → windowing → PSD → tremor-band power → dominant frequency → RMS/amplitude → power ratio. Every stage is testable offline against recorded or synthetic data, independent of hardware.

### 10 Raw Data Loader (Offline)

**Goal:** Load a recorded `raw_stream.csv` into a NumPy array with the documented shape/unit conventions, decoupling all further signal-processing work from live hardware.

**Implementation:**
- Implement a loader function (in `signal_processing/calibration.py` or a small shared loader module — exact filename TBD, follow `architecture.md` folder structure) that reads the CSV and returns `timestamps_us: NDArray[np.int64]` and `raw_signal: NDArray[np.int16]` of shape `(n_samples, 6)` (ax, ay, az, gx, gy, gz), still in raw LSB units.

**Inputs / Outputs:**
- Input: `data/raw/<subject>/<session>/raw_stream.csv`.
- Output: `(timestamps_us, raw_signal)` NumPy arrays, raw LSB units.

**Verification (offline signal-processing test):**
- `pytest tests/test_loader.py`: load the CSV recorded in Feature 09; assert returned array shape is `(n_samples, 6)`, dtype matches, and `n_samples` matches the row count of the source file.

---

### 11 Sensor Calibration (Offset Estimation)

**Goal:** Implement `signal_processing/calibration.py` to estimate accelerometer/gyroscope offsets from a stationary recording and convert raw LSB values to g/°/s.

**Implementation:**
- Implement offset estimation: average raw LSB values over a stationary recording, per axis.
- Implement `apply_calibration(raw_signal, offsets) -> NDArray[np.float64]` converting to g (accel) and °/s (gyro) using the configured range scale factors (`config.sensor.accel_range_g`, `config.sensor.gyro_range_dps`).

**Inputs / Outputs:**
- Input: raw LSB signal `(n_samples, 6)`, stationary calibration recording.
- Output: calibrated signal `(n_samples, 6)`, units g/°/s; estimated offset vector `(6,)`.

**Verification (offline signal-processing test):**
- `pytest tests/test_calibration.py` using the stationary recording from Feature 09: after calibration, mean gyro values must be near-zero (within an experimentally acceptable tolerance), and mean accel z-axis must be near 1 g (device flat) — confirms the "stationary device produces stable acceleration and near-zero gyroscope readings" check from `architecture.md` → Core User Flow.

---

### 12 Axis Handling Strategies

**Goal:** Implement `signal_processing/axis_handling.py` supporting per-axis, 3-axis magnitude, and strongest-tremor-axis representations, selectable via `config.signal.axis_strategy`.

**Implementation:**
- Implement `compute_magnitude(signal) -> NDArray` (per `library-docs.md` → NumPy example).
- Implement `select_strongest_axis(signal, tremor_band_hz, sample_rate_hz) -> NDArray` (selects the axis with highest tremor-band power — depends on Feature 15's filtering/PSD, so this specific sub-function is deferred and stubbed here; the per-axis and magnitude paths are implemented now).
- Implement a dispatcher `get_axis_representation(signal, strategy: str) -> NDArray` reading `config.signal.axis_strategy`.

**Inputs / Outputs:**
- Input: calibrated signal `(n_samples, 6)` or `(n_samples, 3)` per sensor type.
- Output: `(n_samples,)` for magnitude/strongest-axis, or `(n_samples, 3)` for per-axis.

**Verification (offline signal-processing test):**
- `pytest tests/test_axis_handling.py`: on a synthetic 3-axis signal with known per-axis energy, confirm `compute_magnitude` output shape and values match `np.linalg.norm` reference computation.

---

### 13 Gravity / Baseline Removal

**Goal:** Remove the gravity/DC baseline component from the calibrated accelerometer signal.

**Implementation:**
- Implement a high-pass or mean-subtraction baseline-removal function in `signal_processing/filtering.py` (module also to host band-pass filtering, Feature 14), operating on the calibrated signal from Feature 11.

**Inputs / Outputs:**
- Input: calibrated accel signal `(n_samples, 3)` or axis representation from Feature 12, units g.
- Output: baseline-removed signal, same shape/units, mean near zero.

**Verification (offline signal-processing test):**
- `pytest tests/test_baseline_removal.py`: on the stationary recording, confirm output mean is near zero (within tolerance); on a synthetic signal with a known DC offset added, confirm the offset is removed.

---

### 14 Band-Pass Filtering (Tremor + Voluntary Bands)

**Goal:** Implement Butterworth band-pass filtering for the tremor band (4–12 Hz) and voluntary-movement band (0.5–4 Hz) per `library-docs.md` → SciPy.

**Implementation:**
- Implement `bandpass_filter(signal, sample_rate_hz, band_hz, order) -> NDArray` in `signal_processing/filtering.py` using `scipy.signal.butter` + `filtfilt`, reading band edges/order from config.
- Explicitly document (docstring) that this is the offline/non-causal implementation, per `code-standards.md` → Signal Processing.

**Inputs / Outputs:**
- Input: baseline-removed signal `(n_samples,)`, units g/°/s.
- Output: filtered signal, same shape/units.

**Verification (offline signal-processing test):**
- `pytest tests/test_filtering.py`: apply the filter to a synthetic signal composed of a known 6 Hz sine (should pass, tremor band) summed with a known 1 Hz sine (should pass, voluntary band, if testing that filter) and a known 30 Hz sine (should be attenuated); assert the 6 Hz component's amplitude is preserved (within tolerance) and the 30 Hz component is attenuated by an expected margin based on filter order/rolloff.

---

### 15 Complete Feature 12 — Strongest-Axis Selection

**Goal:** Complete the strongest-tremor-axis selection stub from Feature 12, now that filtering (Feature 14) is available.

**Implementation:**
- Implement `select_strongest_axis()` fully: band-pass filter each axis (tremor band), compute power per axis, return the highest-power axis's signal.

**Inputs / Outputs:**
- Input: calibrated multi-axis signal `(n_samples, 3)`.
- Output: `(n_samples,)` single-axis signal, the strongest tremor axis.

**Verification (offline signal-processing test):**
- `pytest tests/test_axis_handling.py` (extended): on a synthetic 3-axis signal where one axis has a strong 6 Hz component and the others are near-zero, confirm the function selects the correct axis.

---

### 16 Overlapping Windowing

**Goal:** Implement `signal_processing/windowing.py` to segment a filtered signal into overlapping windows per `config.signal.window_length_s` / `window_overlap_pct`.

**Implementation:**
- Implement `segment_windows(signal, sample_rate_hz, window_length_s, overlap_pct) -> NDArray` producing shape `(n_windows, window_samples, n_axes)`.
- Discard a trailing incomplete window rather than zero-padding, per `code-standards.md` → Error Handling.

**Inputs / Outputs:**
- Input: filtered signal `(n_samples,)` or `(n_samples, n_axes)`.
- Output: `(n_windows, window_samples, n_axes)` (or `(n_windows, window_samples)` for single-axis).

**Verification (offline signal-processing test):**
- `pytest tests/test_windowing.py`: on a signal of known length, confirm `n_windows` matches the expected formula (`(n_samples - window_samples) // step_samples + 1`), confirm 50% overlap produces the expected step size, and confirm no trailing partial window is zero-padded.

---

### 17 Welch PSD Computation

**Goal:** Implement `signal_processing/spectral_analysis.py`'s Welch PSD function per `library-docs.md` → SciPy.

**Implementation:**
- Implement `compute_welch_psd(signal, sample_rate_hz, nperseg) -> (freqs_hz, psd)` using `scipy.signal.welch`.

**Inputs / Outputs:**
- Input: single window `(window_samples,)`.
- Output: `freqs_hz (n_freq_bins,)`, `psd (n_freq_bins,)`.

**Verification (offline signal-processing test):**
- `pytest tests/test_spectral_analysis.py`: on a synthetic 6 Hz sine window, confirm the PSD's peak frequency bin is closest to 6 Hz (within the frequency resolution determined by `nperseg`/sample rate).

---

### 18 Tremor-Band Power, Total Power, Power Ratio, Dominant Frequency

**Goal:** Complete `signal_processing/spectral_analysis.py` with the remaining spectral features.

**Implementation:**
- Implement `tremor_band_power(freqs_hz, psd, tremor_band_hz) -> float` (integrate PSD over the tremor band).
- Implement `total_power(freqs_hz, psd) -> float`.
- Implement `power_ratio(tremor_power, total_power) -> float` with the division-by-zero guard (`+ 1e-12`) per `code-standards.md`.
- Implement `dominant_frequency(freqs_hz, psd, tremor_band_hz) -> float` (peak bin within the tremor band).

**Inputs / Outputs:**
- Input: `freqs_hz`, `psd` from Feature 17.
- Output: four scalar features per window.

**Verification (offline signal-processing test):**
- `pytest tests/test_spectral_analysis.py` (extended): on the synthetic 6 Hz window, confirm `dominant_frequency` ≈ 6 Hz, `tremor_band_power` is the dominant contributor to `total_power`, and `power_ratio` is close to 1.0 for a pure in-band signal.

---

### 19 Time-Domain Features (RMS, Variance)

**Goal:** Implement `signal_processing/time_domain.py` for RMS/amplitude and variance.

**Implementation:**
- Implement `compute_rms(signal) -> float` per `library-docs.md` → NumPy example.
- Implement `compute_variance(signal) -> float` using `np.var`.

**Inputs / Outputs:**
- Input: single window, filtered tremor-band signal `(window_samples,)`.
- Output: `rms_amplitude: float`, `variance: float`.

**Verification (offline signal-processing test):**
- `pytest tests/test_time_domain.py`: on a synthetic sine of known amplitude `A`, confirm `compute_rms` ≈ `A / sqrt(2)` within tolerance.

---

### 20 Feature Vector Assembly

**Goal:** Implement `signal_processing/feature_extraction.py` to assemble the complete documented feature vector for one window.

**Implementation:**
- Implement `extract_features(window, sample_rate_hz) -> dict` returning `{tremor_band_power, total_power, power_ratio, dominant_frequency_hz, rms_amplitude, variance, spectral_entropy, accel_magnitude, gyro_magnitude}`, calling Features 17–19's functions plus a spectral-entropy computation and axis-magnitude computation (reusing Feature 12).
- Add `subject_id`, `session_id`, `window_id` metadata fields.

**Inputs / Outputs:**
- Input: one window `(window_samples, n_axes)` plus metadata.
- Output: feature dict matching the exact schema in `architecture.md` → Interfaces & Data Contracts.

**Verification (offline signal-processing test):**
- `pytest tests/test_feature_extraction.py`: confirm the returned dict has exactly the documented keys (no missing, no extra), correct types, and plausible values on the synthetic 6 Hz window.

---

### 21 Milestone 2 Verification Script — Raw vs. Filtered Visual Check

**Goal:** Produce the Milestone-2 deliverable from `architecture.md` → Signal-Processing Pipeline: visually verify raw vs. filtered signal and tremor spectrum on recorded data.

**Implementation:**
- Create `scripts/run_pipeline_baseline.py`: loads a recorded session (Feature 09/10), runs calibration → axis handling → baseline removal → filtering → windowing → PSD (Features 11–18), and plots raw vs. filtered signal and PSD using matplotlib.

**Inputs / Outputs:**
- Input: `data/raw/<subject>/<session>/raw_stream.csv`.
- Output: plots (saved to `data/processed/<subject>/<session>/` or shown interactively — exact output path TBD), and optionally `data/processed/<subject>/<session>/windows.parquet`.

**Verification (offline signal-processing test):**
- Run the script against a recorded session with a known/simulated tremor-like hand motion (e.g., a deliberate ~5–6 Hz wrist shake). Visually confirm: the filtered signal shows a clear oscillation at the shake frequency, and the PSD shows a distinct peak within the 4–12 Hz tremor band. This satisfies `architecture.md` Milestone 2 before any ML work begins.

---

## Phase 4 — Tremor Detection and ML

Build feature extraction (already complete via Phase 3), dataset preparation, model training, evaluation, and inference — strictly in that order, with subject-level splitting enforced throughout.

### 22 Dataset Builder with Subject-Level Splitting

**Goal:** Implement `ml/dataset_builder.py` to assemble a labeled feature table and produce subject-level train/validation/test splits.

**Implementation:**
- Implement `build_feature_table(feature_records) -> pd.DataFrame` per `library-docs.md` → pandas, validating required columns.
- Implement `get_subject_level_folds(X, y, groups, n_splits) -> Generator` using `sklearn.model_selection.GroupKFold`, per `library-docs.md` → scikit-learn.
- Add a label-assignment step: recordings must be tagged tremor/no-tremor at the session or window level (mechanism TBD — depends on how recordings are collected; document as an Open Item if not yet defined).

**Inputs / Outputs:**
- Input: multiple `features.csv` files across subjects/sessions (from Feature 20 applied to multiple recordings).
- Output: assembled `pd.DataFrame`, and a generator of `(train_idx, test_idx)` index pairs grouped by `subject_id`.

**Verification (ML evaluation test):**
- `pytest tests/test_dataset_builder.py`: construct a synthetic feature table with a known set of `subject_id`s; assert that no `subject_id` appears in both the train and test index sets of any fold produced by `get_subject_level_folds` (explicit leakage check).

---

### 23 Baseline Signal-Processing Threshold Detector

**Goal:** Implement a non-ML baseline detector (per `architecture.md` → Development Philosophy: "establish a non-ML tremor detector" before comparing ML models).

**Implementation:**
- Implement a simple threshold rule (e.g., `tremor_band_power > threshold` or `power_ratio > threshold`) as a baseline classifier function, location TBD within `ml/` (e.g., `ml/baseline_detector.py`).

**Inputs / Outputs:**
- Input: feature vector (Feature 20 output).
- Output: `label: bool`.

**Verification (ML evaluation test):**
- `pytest tests/test_baseline_detector.py`: on the labeled dataset from Feature 22, compute precision/recall/F1 for the threshold rule using `sklearn.metrics`; record the result as the baseline to beat in later model comparison. No specific target score is asserted yet — this establishes the reference point.

---

### 24 Candidate ML Model Training

**Goal:** Implement `ml/train.py` to fit Logistic Regression, SVM, and Random Forest candidates with proper scaling.

**Implementation:**
- Implement `train_and_scale(X_train, y_train)` per `library-docs.md` → scikit-learn: fit `StandardScaler` on training data only, fit all three candidate models with `random_state=config.ml.random_seed`.
- Persist each fitted model and the scaler using `joblib.dump`, following the naming convention `model_<algorithm>_<version>.pkl` / `scaler_<version>.pkl` from `code-standards.md` → File and Folder Naming.

**Inputs / Outputs:**
- Input: `X_train, y_train` from a Feature 22 fold.
- Output: `data/models/model_logistic_regression_v1.pkl`, `model_svm_v1.pkl`, `model_random_forest_v1.pkl`, `data/models/scaler_v1.pkl`.

**Verification (ML evaluation test):**
- `pytest tests/test_train.py`: confirm all three models fit without error on a synthetic labeled dataset and that saved artifact files exist at the expected paths after `train_and_scale` + `joblib.dump`.

---

### 25 Model Evaluation and Comparison

**Goal:** Implement `ml/evaluate.py` to report precision, recall, F1, sensitivity, and false-positive rate per candidate model using subject-level cross-validation.

**Implementation:**
- Implement `evaluate_model(model, scaler, X_test, y_test) -> dict` returning the required metrics via `sklearn.metrics`.
- Run evaluation across all `GroupKFold` folds from Feature 22 for each of the three candidate models plus the Feature 23 baseline.

**Inputs / Outputs:**
- Input: fitted models (Feature 24), test folds (Feature 22).
- Output: a metrics table (per model, per fold, and aggregated) — format TBD (CSV/JSON under `data/models/` or a report location).

**Verification (ML evaluation test):**
- `pytest tests/test_evaluate.py`: confirm `evaluate_model` returns all five required metrics as floats in `[0, 1]` on a synthetic dataset with known TP/FP/FN/TN counts, matching hand-computed expected values.

---

### 26 Smallest-Model Selection for ESP32

**Goal:** Implement `ml/model_selection.py` to select the smallest model meeting the project's detection requirements, per `architecture.md` → ML Architecture.

**Implementation:**
- Implement `select_model(evaluation_results, min_recall, min_precision) -> str` (returns the selected model name), preferring the smallest artifact size among models meeting the thresholds. Exact `min_recall`/`min_precision` thresholds are TBD (experimental) — read from config once determined, or explicitly flagged as unset until then.

**Inputs / Outputs:**
- Input: evaluation results from Feature 25, model file sizes.
- Output: selected model name/version, recorded in a small selection report.

**Verification (ML evaluation test):**
- `pytest tests/test_model_selection.py`: given a synthetic evaluation-results table with known metrics and file sizes, confirm `select_model` returns the smallest model that meets the given thresholds, and returns none/raises clearly if no candidate meets them.

---

### 27 ML Inference Module

**Goal:** Implement `ml/inference.py` — the sole entry point the controller will use for detection/severity/confidence.

**Implementation:**
- Implement `predict(features: dict, model_path, scaler_path) -> InferenceResult` per `library-docs.md` → scikit-learn: enforce exact feature ordering, load model/scaler via `joblib`, never re-fit the scaler.
- Implement severity estimation (rule-based on `tremor_band_power`/`rms_amplitude`, or model-derived — mechanism TBD, document choice once made) behind the same function's return contract.
- Return the full documented ML output contract: `{label, severity, confidence, dominant_frequency_hz, amplitude, phase: None}` (frequency/amplitude populated via Phase 5 functions once available; `phase` remains `None` until Phase 5's phase estimation, if implemented).

**Inputs / Outputs:**
- Input: a feature dict (Feature 20 output) for one window.
- Output: `InferenceResult` matching `architecture.md`'s ML output contract exactly.

**Verification (ML evaluation test):**
- `pytest tests/test_inference.py`: run `predict()` on the recorded session from Feature 21 (known to contain deliberate tremor-like motion) using the model selected in Feature 26; confirm `label=True` during the tremor-motion segment and `confidence` is a float in `[0, 1]`.

---

## Phase 5 — Frequency, Amplitude, and Phase Estimation

Implement frequency and amplitude estimation first, since both are required for the initial controller (Phase 7). Phase estimation is a separate, later-stage feature and must not block controller development.

### 28 Dominant Frequency Estimation Module

**Goal:** Implement `estimation/frequency_estimation.py`, wrapping Feature 18's dominant-frequency logic behind the `estimation/` module boundary defined in `architecture.md`.

**Implementation:**
- Implement `estimate_frequency(freqs_hz, psd, tremor_band_hz) -> float | None`, reusing `signal_processing/spectral_analysis.py`'s PSD output but returning `None` if no clear peak exists (flat spectrum), per `code-standards.md` → Error Handling.

**Inputs / Outputs:**
- Input: `freqs_hz`, `psd` for one window.
- Output: `dominant_frequency_hz: float | None`.

**Verification (offline signal-processing test):**
- `pytest tests/test_frequency_estimation.py`: on the synthetic 6 Hz window, confirm output ≈ 6 Hz; on a flat/noise-only PSD, confirm output is `None`.

---

### 29 Amplitude Estimation Module

**Goal:** Implement `estimation/amplitude_estimation.py`, wrapping Feature 19's RMS logic behind the `estimation/` module boundary.

**Implementation:**
- Implement `estimate_amplitude(filtered_signal) -> float` calling `signal_processing/time_domain.py`'s `compute_rms`.

**Inputs / Outputs:**
- Input: filtered tremor-band signal for one window.
- Output: `amplitude: float`, units g.

**Verification (offline signal-processing test):**
- `pytest tests/test_amplitude_estimation.py`: same synthetic-sine assertion as Feature 19, invoked through the `estimation/` wrapper.

---

### 30 Frequency/Amplitude Integration into ML Inference Output

**Goal:** Wire Features 28–29 into `ml/inference.py`'s output so the full documented ML output contract is populated (previously `phase=None` only; now frequency/amplitude are real values).

**Implementation:**
- Update `predict()` (Feature 27) to call `estimate_frequency()` and `estimate_amplitude()` on the same window used for classification, populating `dominant_frequency_hz` and `amplitude` in the returned `InferenceResult`.

**Inputs / Outputs:**
- Input: window signal + PSD (already computed upstream).
- Output: complete `InferenceResult` with real frequency/amplitude values.

**Verification (ML evaluation test):**
- `pytest tests/test_inference.py` (extended): on the Feature 21 recorded tremor session, confirm `dominant_frequency_hz` falls within the 4–12 Hz tremor band during the tremor-motion segment, and `amplitude` is a positive float.

---

### 31 Phase Estimation (Advanced, Experimental)

**Goal:** Implement `estimation/phase_estimation.py` as a clearly separate, optional, disabled-by-default module, per `architecture.md` → Frequency & Phase Estimation.

**Implementation:**
- Implement an initial phase-estimation approach (method TBD — e.g., Hilbert-transform-based instantaneous phase from the filtered tremor-band signal), returning `phase: float | None`.
- Gate this feature behind an explicit config flag (e.g., `config.estimation.phase_enabled`, to be added to `system_config.yaml` if not already present) defaulting to disabled.

**Inputs / Outputs:**
- Input: filtered tremor-band window signal.
- Output: `phase: float | None`.

**Verification (offline signal-processing test):**
- `pytest tests/test_phase_estimation.py`: on a synthetic sine of known phase, confirm the estimated phase is within an experimentally acceptable error margin (margin TBD, record measured error as the baseline). This verification must pass before phase is ever consumed by controller logic — per Build Order Rules, phase-aware control is not built until this is validated.

---

## Phase 6 — Stimulation Simulation

Build the simulated stimulation-response subsystem, fully decoupled from the controller, before any adaptive-control code exists. Simulation-only throughout — no physical actuation.

### 32 Synthetic Tremor Signal Generator

**Goal:** Implement `simulation/tremor_model.py`'s synthetic tremor generator, providing controllable ground-truth tremor signals for simulation and validation.

**Implementation:**
- Implement `generate_synthetic_tremor(frequency_hz, amplitude, duration_s, sample_rate_hz, noise_std) -> NDArray` producing a sine-based tremor signal with configurable frequency/amplitude/noise, usable both as simulation input and as ground truth for frequency/amplitude-accuracy validation (Phase 9).

**Inputs / Outputs:**
- Input: frequency, amplitude, duration, sample rate, noise level.
- Output: synthetic signal `(n_samples,)`, units g.

**Verification (simulation test):**
- `pytest tests/test_tremor_model.py`: confirm the generated signal's dominant frequency (via Feature 28's estimator) matches the requested `frequency_hz` within tolerance, and RMS amplitude (via Feature 29) matches the requested `amplitude` within tolerance.

---

### 33 Simulated Stimulation-Response Model

**Goal:** Implement `simulation/stimulation_model.py`'s ODE/state-space model relating `StimParams` to tremor suppression, per `library-docs.md` → MATLAB/Simulink OR Python-based simulation.

**Implementation:**
- Implement the ODE right-hand side and `scipy.integrate.solve_ivp` call per the pattern in `library-docs.md`, with `max_step` tied to `config.simulation.timestep_s`.
- Include the required module-level docstring stating this is not a clinically validated physiological model.
- Implement latency modeling (`config.simulation.latency_ms`) as an explicit delay applied to the stimulation effect, not merely reported.

**Inputs / Outputs:**
- Input: `StimParams` (Feature 03's dataclass), initial tremor state.
- Output: post-mitigation tremor signal trajectory.

**Verification (simulation test):**
- `pytest tests/test_stimulation_model.py`: with a fixed, non-trivial `StimParams` and a synthetic input tremor (Feature 32), confirm the ODE solve completes without raising, `stability_warning=False` under nominal parameters, and increasing `amplitude` in `StimParams` (within bounds) produces a monotonically greater suppression effect (sanity check on model direction, not exact clinical accuracy).

---

### 34 Suppression Measurement

**Goal:** Implement suppression-percentage computation comparing pre- and post-mitigation tremor signals.

**Implementation:**
- Implement `compute_suppression_pct(pre_signal, post_signal) -> float` (e.g., based on tremor-band power reduction, reusing Feature 18's `tremor_band_power`), located in `simulation/` or `validation/metrics.py` per architecture (exact placement: `simulation/` for the real-time-loop-facing version, `validation/metrics.py` may reuse it for reporting).

**Inputs / Outputs:**
- Input: pre-mitigation and post-mitigation signal windows.
- Output: `achieved_suppression_pct: float`.

**Verification (simulation test):**
- `pytest tests/test_suppression_measurement.py`: with a synthetic pre-signal of known power and a synthetic post-signal at a known fraction of that power, confirm `compute_suppression_pct` returns the expected percentage within tolerance.

---

### 35 Simulation Interface — `simulation.apply()`

**Goal:** Assemble Features 32–34 behind the single documented entry point `simulation.apply(params: StimParams, tremor_state) -> SimResult`.

**Implementation:**
- Implement `apply()` in `simulation/stimulation_model.py` (or a small `simulation/__init__.py`/facade), returning the full `SimResult` contract: `{post_mitigation_signal, achieved_suppression_pct, residual_amplitude, latency_ms, stability_warning}`.
- Validate incoming `StimParams` against `config.controller.param_bounds` at this boundary too (defense-in-depth per `code-standards.md`), raising `ValueError` on out-of-bounds input.

**Inputs / Outputs:**
- Input: `StimParams`, `tremor_state`.
- Output: `SimResult`.

**Verification (simulation test):**
- `pytest tests/test_simulation_apply.py`: call `apply()` with valid bounded `StimParams` and confirm a well-formed `SimResult` is returned; call it with an out-of-bounds `StimParams` and confirm `ValueError` is raised.

---

## Phase 7 — Adaptive Controller

Implement the closed-loop controller only now that detection (Phase 4), frequency/amplitude analysis (Phase 5), and the simulation interface (Phase 6) are independently verified.

### 36 Confidence Gating and Mitigation Decision Logic

**Goal:** Implement `controller/decision_logic.py`'s mitigate/no-mitigate decision with confidence gating.

**Implementation:**
- Implement `decide_mitigation(ml_output: InferenceResult, config) -> bool`: force `False` if `confidence < config.ml.confidence_threshold`; otherwise apply `severity >= config.controller.severity_threshold` with hysteresis enter/exit thresholds (`config.controller.hysteresis_pct`) tracked via `ControllerState`.

**Inputs / Outputs:**
- Input: `InferenceResult` (Feature 30 output), `ControllerState`.
- Output: `mitigate: bool`, updated `ControllerState`.

**Verification (offline controller test):**
- `pytest tests/test_decision_logic.py`: with `confidence` below threshold and high `severity`, confirm `mitigate=False`; with `confidence` above threshold and `severity` above the mitigation threshold, confirm `mitigate=True`; confirm hysteresis prevents rapid toggling across a threshold-straddling synthetic severity sequence.

---

### 37 Bounded Stimulation Parameter Selection

**Goal:** Implement `controller/parameter_selection.py`'s initial `StimParams` selection when `mitigate=True`.

**Implementation:**
- Implement `select_initial_params(ml_output, config) -> StimParams`: choose a starting point within `config.controller.param_bounds`, informed by `severity`/`dominant_frequency_hz` per `architecture.md` → Adaptive Controller. `phase` remains `None` unless Feature 31 is enabled and validated.

**Inputs / Outputs:**
- Input: `InferenceResult`.
- Output: `StimParams`, within configured bounds.

**Verification (offline controller test):**
- `pytest tests/test_parameter_selection.py`: for a range of synthetic `severity` values, confirm every returned `StimParams` field is within `config.controller.param_bounds`.

---

### 38 Adaptation Logic (Increase/Decrease/Maintain)

**Goal:** Implement `controller/adaptation.py`'s feedback-based parameter adjustment, capped and hysteresis-gated.

**Implementation:**
- Implement `adapt_params(current_params, achieved_suppression_pct, target_suppression_pct, config, state) -> (StimParams, ControllerState)`: increase/decrease/maintain per `architecture.md` → Adaptive Controller, capped by `config.controller.max_delta_per_step`, clamped to bounds with `bound_clamped=True` logging on clamp.
- On invalid input (`achieved_suppression_pct=None` or simulation `stability_warning=True`), default to `maintain`, per `code-standards.md`.

**Inputs / Outputs:**
- Input: current `StimParams`, `achieved_suppression_pct`, `ControllerState`.
- Output: updated `StimParams`, updated `ControllerState`.

**Verification (offline controller test):**
- `pytest tests/test_adaptation.py`: with `achieved_suppression_pct` below target, confirm parameters increase (bounded by `max_delta_per_step`); above target, confirm decrease (minimizing exposure); within the hysteresis band, confirm maintain; with `stability_warning=True`, confirm maintain regardless of suppression value.

---

### 39 Controller State Management

**Goal:** Implement `controller/controller_state.py`'s `ControllerState` dataclass and ensure it is threaded explicitly (never a global) through Features 36–38.

**Implementation:**
- Implement `ControllerState{current_params, hysteresis_active, adaptation_history, last_decision_timestamp}` per `code-standards.md` → Module and Component Structure template.

**Inputs / Outputs:**
- Output: a single explicit state object passed in/out of every controller function call.

**Verification (offline controller test):**
- `pytest tests/test_controller_state.py`: run a sequence of `decide_mitigation` → `select_initial_params`/`adapt_params` calls, threading `ControllerState` explicitly, and confirm no controller function relies on any module-level mutable state (verified by asserting a fresh `ControllerState` produces identical output to two independent runs given identical inputs — determinism check).

---

### 40 Full Controller Loop (Detect → Decide → Mitigate, Single Cycle)

**Goal:** Assemble Features 36–39 into one function representing a single controller cycle, consuming ML output and producing `StimParams` — without yet calling the simulation (that is Feature 41).

**Implementation:**
- Implement `run_controller_cycle(ml_output: InferenceResult, state: ControllerState, config) -> (StimParams | None, ControllerState)` in `controller/` (facade function, exact file TBD — likely alongside `controller_state.py` or a new `controller/__init__.py`).

**Inputs / Outputs:**
- Input: `InferenceResult`.
- Output: `StimParams` (or `None` if not mitigating), updated `ControllerState`.

**Verification (offline controller test):**
- `pytest tests/test_controller_cycle.py`: feed a sequence of synthetic `InferenceResult`s (varying confidence/severity) and confirm the sequence of decisions/parameters matches expected behavior derived from Features 36–38's individually-verified logic — an integration test of the controller alone, still without simulation.

---

## Phase 8 — Closed-Loop Integration

Connect all previously verified components (acquisition or recorded data → signal processing → ML/estimation → controller → simulation → suppression measurement → adaptation) into the complete loop, without bypassing any documented interface.

### 41 Closed-Loop Runner (Single-Cycle, Simulated Tremor Input)

**Goal:** Implement `simulation/closed_loop_runner.py` to wire the controller (Phase 7) to the simulation (Phase 6) for one full Detect→Adapt cycle, using a synthetic tremor input (Feature 32) rather than recorded hardware data.

**Implementation:**
- Implement `run_closed_loop_cycle(tremor_signal, state: ControllerState, config) -> (SimResult, ControllerState)`: run Features 20 (feature extraction) → 27/30 (ML inference with frequency/amplitude) → 40 (controller cycle) → 35 (`simulation.apply`) → 34 (suppression measurement) → 38 (adaptation) in sequence, each stage receiving only the documented interface output of the previous stage.

**Inputs / Outputs:**
- Input: a synthetic or recorded tremor signal window.
- Output: `SimResult`, updated `ControllerState`.

**Verification (closed-loop test):**
- `pytest tests/test_closed_loop_single_cycle.py`: run one cycle on a synthetic 6 Hz, above-threshold-severity tremor signal; confirm the full chain executes without error, `mitigate=True` was decided, `StimParams` were applied to the simulation, and `achieved_suppression_pct > 0`.

---

### 42 Multi-Cycle Closed-Loop Simulation

**Goal:** Run the closed loop across multiple successive cycles/windows, verifying that `ControllerState` persists correctly and adaptation converges toward the target suppression.

**Implementation:**
- Create `scripts/run_closed_loop_simulation.py`: repeatedly calls `run_closed_loop_cycle()` across N windows of a synthetic (and later recorded) tremor signal, logging `ControllerState`, `StimParams`, and `SimResult` per cycle.

**Inputs / Outputs:**
- Input: a longer synthetic tremor signal (e.g., 60 seconds at constant frequency/amplitude).
- Output: per-cycle log (`data/simulation/<experiment_id>/cycle_log.csv` or similar — exact path TBD), a suppression-vs-time plot.

**Verification (closed-loop test):**
- Run the script on a synthetic constant-tremor input. Confirm `achieved_suppression_pct` trends toward `config.controller.target_suppression_pct` over successive cycles (not oscillating without bound), and that `StimParams` remain within bounds throughout — this is the first end-to-end demonstration of the full Detect → Analyze → Decide → Mitigate → Measure → Adapt loop from `architecture.md`.

---

### 43 Closed-Loop with Recorded Hardware Data

**Goal:** Re-run the multi-cycle closed loop (Feature 42) using a real recorded ESP32/MPU6050 session (Feature 09/21) instead of purely synthetic input, confirming the loop works on real sensor data end-to-end (still simulated stimulation).

**Implementation:**
- Extend `scripts/run_closed_loop_simulation.py` with a `--source recorded --path <raw_stream.csv>` mode that runs the full pipeline from Feature 10 (raw loader) through Feature 41's closed loop.

**Inputs / Outputs:**
- Input: `data/raw/<subject>/<session>/raw_stream.csv` containing a deliberate tremor-like motion segment.
- Output: same per-cycle log/plot as Feature 42, generated from real sensor data.

**Verification (closed-loop test):**
- Confirm the loop correctly detects the deliberate-motion segment (`mitigate=True` during that segment, `False` outside it) and produces a plausible suppression trend during the tremor segment — first confirmation that the entire documented data flow in `architecture.md` works end-to-end on real hardware-sourced data.

---

## Phase 9 — Validation and Robustness

Build the validation/experimentation infrastructure, strictly reusing production controller/simulation code (never a forked copy), per `architecture.md` → Validation Architecture.

### 44 No-Mitigation Baseline Experiment

**Goal:** Implement `validation/experiments/no_mitigation.py`.

**Implementation:**
- Run the closed-loop pipeline with the controller's mitigation decision forced to `False` for every cycle, recording tremor power over time with no intervention.

**Inputs / Outputs:**
- Input: synthetic and/or recorded tremor signal.
- Output: `data/validation/no_mitigation_<run_id>/report.json` (tremor power trace, no suppression).

**Verification (end-to-end / validation test):**
- Confirm the recorded tremor power remains statistically unchanged over the run (no suppression trend), establishing the baseline for comparison.

---

### 45 Fixed-Parameter Mitigation Experiment

**Goal:** Implement `validation/experiments/fixed_mitigation.py`.

**Implementation:**
- Run the closed-loop pipeline with a static, non-adapting `StimParams` (bypassing Feature 38's `adapt_params`), recording suppression achieved with a fixed strategy.

**Inputs / Outputs:**
- Output: `data/validation/fixed_mitigation_<run_id>/report.json`.

**Verification (end-to-end / validation test):**
- Confirm suppression is achieved (`achieved_suppression_pct > 0` consistently) but does not adapt to varying tremor severity across the run (flat parameter trace by construction).

---

### 46 Adaptive Mitigation Experiment

**Goal:** Implement `validation/experiments/adaptive_mitigation.py`, running the full unmodified production controller.

**Implementation:**
- Reuse Feature 42's multi-cycle closed-loop runner unmodified, writing results to the validation report format.

**Inputs / Outputs:**
- Output: `data/validation/adaptive_mitigation_<run_id>/report.json`.

**Verification (end-to-end / validation test):**
- Confirm the adaptive run achieves the target suppression using a lower total simulated stimulation exposure/duty cycle than the fixed-parameter run (Feature 45) on the same input tremor signal — the core success criterion from `architecture.md`.

---

### 47 Validation Metrics Module

**Goal:** Implement `validation/metrics.py`, consolidating all comparison metrics computed across Features 44–46.

**Implementation:**
- Implement functions for: tremor-power reduction, residual amplitude, time-to-target, duty cycle, total simulated exposure, false-activation rate, detection-to-actuation latency, controller stability (parameter oscillation measure).

**Inputs / Outputs:**
- Input: per-cycle logs from Features 44–46.
- Output: a consolidated comparison table (`data/validation/comparison_<run_id>/report.json`).

**Verification (end-to-end / validation test):**
- `pytest tests/test_metrics.py`: on synthetic per-cycle logs with known suppression/exposure values, confirm each metric function returns the hand-computed expected value.

---

### 48 Detection Performance and Voluntary-Movement Rejection

**Goal:** Extend `validation/` with detection-specific metrics using recorded data containing both tremor-like motion and voluntary movement.

**Implementation:**
- Reuse Feature 25's evaluation functions against a recorded dataset explicitly containing labeled voluntary-movement segments (0.5–4 Hz band) alongside tremor segments (4–12 Hz band).

**Inputs / Outputs:**
- Input: labeled recorded dataset with tremor and voluntary-movement segments.
- Output: precision/recall/F1/false-positive-rate specifically on the voluntary-movement-rejection sub-task.

**Verification (ML evaluation test):**
- Confirm the false-positive rate on voluntary-movement-only segments is reported and reviewed; no fixed numeric target is asserted here since acceptable thresholds are experimentally determined (per `architecture.md` Success Criteria #3).

---

### 49 Frequency and Amplitude Estimation Accuracy

**Goal:** Quantify frequency/amplitude estimation error against known synthetic ground truth.

**Implementation:**
- Use Feature 32's synthetic tremor generator across a sweep of known frequencies/amplitudes; run Features 28–29's estimators; compute error statistics.

**Inputs / Outputs:**
- Input: synthetic tremor sweep (known frequency/amplitude).
- Output: `data/validation/frequency_amplitude_accuracy_<run_id>/report.json`.

**Verification (offline signal-processing test):**
- Confirm mean/max estimation error across the sweep is recorded; compare against the "experimentally defined acceptable error" referenced in `architecture.md` Success Criteria #4 once that threshold is set (currently TBD).

---

### 50 Phase Accuracy (Conditional — Only If Feature 31 Enabled)

**Goal:** Quantify phase-estimation error, only if Feature 31 (Phase Estimation) has been implemented and enabled.

**Implementation:**
- Same sweep methodology as Feature 49, applied to `estimation/phase_estimation.py`.

**Inputs / Outputs:**
- Output: phase-error report, same format as Feature 49.

**Verification (offline signal-processing test):**
- Confirm phase error is measured and recorded. If Feature 31 remains disabled, skip this feature entirely — do not fabricate phase-aware controller behavior without this verification passing first, per Build Order Rules.

---

### 51 Robustness Testing

**Goal:** Implement `validation/robustness_tests.py` covering sensor noise, varying tremor frequency/amplitude, filtering delay, and deliberately incorrect/low-confidence detections.

**Implementation:**
- Add configurable noise injection to Feature 32's synthetic generator; sweep tremor frequency/amplitude across the closed loop (Feature 42); measure filtering-induced delay by comparing input/output signal timing; deliberately feed artificially low-confidence `InferenceResult`s into the controller to confirm gating behavior (reusing Feature 36's test scenarios at the full closed-loop level).

**Inputs / Outputs:**
- Input: synthetic signals with controlled noise/frequency/amplitude variation, plus deliberately corrupted `InferenceResult`s.
- Output: `data/validation/robustness_<run_id>/report.json`.

**Verification (closed-loop test):**
- Confirm the controller never exceeds parameter bounds, never oscillates outside the hysteresis band, and correctly withholds mitigation on deliberately low-confidence input across the full robustness sweep — directly verifying `architecture.md` Success Criterion #9.

---

## Phase 10 — Visualization and Monitoring

Build observational visualizations around data/interfaces that already exist from Phases 1–9. Read-only throughout; never a pipeline dependency.

### 52 Raw and Filtered Signal Plots

**Goal:** Implement `visualization/signal_plots.py` for raw vs. filtered signal and PSD visualization, formalizing the ad hoc plotting from Feature 21.

**Implementation:**
- Implement `plot_raw_vs_filtered(raw_signal, filtered_signal, sample_rate_hz)` and `plot_psd(freqs_hz, psd, tremor_band_hz)` using matplotlib.

**Inputs / Outputs:**
- Input: signal arrays / PSD arrays from `signal_processing/`.
- Output: matplotlib figures (displayed or saved).

**Verification (offline signal-processing test):**
- Run against the Feature 21 recorded session and visually confirm plots render correctly with correct axis labels/units.

---

### 53 Detection and Severity Dashboard Plot

**Goal:** Visualize ML detection history (label, severity, confidence) over time.

**Implementation:**
- Implement a time-series plot in `visualization/` (module TBD — extend `signal_plots.py` or a new file) consuming a sequence of `InferenceResult`s from a recorded session run through Feature 27/30.

**Inputs / Outputs:**
- Input: sequence of `InferenceResult` over a session.
- Output: plot showing label/severity/confidence vs. time.

**Verification (offline signal-processing test):**
- Run on the Feature 21 recorded session; visually confirm detection correctly aligns with the known deliberate-motion segment.

---

### 54 Controller Dashboard

**Goal:** Implement `visualization/controller_dashboard.py` showing controller state, decision, and `StimParams` over time, per `architecture.md`'s `/controller` functional view.

**Implementation:**
- Implement a plot/report consuming the per-cycle log produced by Feature 42/43 (mitigation decision, target vs. current suppression, `StimParams` fields over cycles).

**Inputs / Outputs:**
- Input: per-cycle closed-loop log.
- Output: multi-panel plot matching the `/controller` view described in `architecture.md` → Pages / Functional Views.

**Verification (offline signal-processing test):**
- Run on the Feature 42/43 output; visually confirm the plot reflects the same adaptation trend already verified numerically in Feature 42/43's tests.

---

### 55 Validation Report Visualization

**Goal:** Implement `visualization/validation_report.py` to render the no-mitigation vs. fixed vs. adaptive comparison from Phase 9.

**Implementation:**
- Implement a summary plot/table consuming `data/validation/comparison_<run_id>/report.json` (Feature 47), matching the `/validation` functional view.

**Inputs / Outputs:**
- Input: Feature 47's consolidated report.
- Output: comparison plot/table (suppression, exposure, duty cycle across the three strategies).

**Verification (offline signal-processing test):**
- Run against Feature 46/47's output and visually confirm the adaptive-vs-fixed exposure comparison matches the numeric result already verified in Feature 46.

---

## Phase 11 — ESP32 Real-Time Deployment

Move only components explicitly intended for on-device execution, per `architecture.md` → Deployment Architecture and Development vs. Production Boundary. Do not port algorithms to the ESP32 speculatively.

### 56 ESP32 Resource Profiling of Selected Model

**Goal:** Assess whether the model selected in Feature 26 fits ESP32 memory/latency constraints, before committing to any on-device port.

**Implementation:**
- Measure the selected model's serialized size and estimated inference latency (measured on PC, scaled/estimated for ESP32 class hardware) against `architecture.md` → Performance & Timing Constraints.

**Inputs / Outputs:**
- Input: selected model artifact (Feature 26).
- Output: a resource-fit report (fits / does not fit / TBD pending real on-device measurement).

**Verification (hardware-dependent, preliminary):**
- Record model size and estimated latency; explicitly mark this feature as inconclusive/TBD if only PC-side estimates are available — do not claim ESP32 fit without an actual on-device measurement.

---

### 57 On-Device Filtering/Windowing Port (Conditional)

**Goal:** Port `signal_processing/filtering.py` and `windowing.py`'s causal equivalents onto the ESP32, only if Feature 56 confirms feasibility and the architecture's migration-candidate designation (per `architecture.md` → Development vs. Production Boundary) is acted on.

**Implementation:**
- Implement a causal (non-`filtfilt`) band-pass filter in C++, per `code-standards.md`'s causal-vs-offline distinction, and a fixed-size circular buffer for windowing on the ESP32.

**Inputs / Outputs:**
- Input: raw samples on-device.
- Output: filtered/windowed signal computed on-device.

**Verification (hardware-dependent):**
- Compare on-device filter output against the PC-side `filtfilt`-based offline filter (Feature 14) on the same recorded input, run through both implementations; confirm the causal on-device output tracks the offline reference within an experimentally acceptable phase-delay-adjusted tolerance.

---

### 58 On-Device Inference Port (Conditional)

**Goal:** Port the selected lightweight model (Feature 26) to run inference on-device, only if Feature 56/57 confirm feasibility.

**Implementation:**
- Export the selected scikit-learn model to a minimal on-device-executable representation (mechanism TBD — e.g., manual coefficient export for Logistic Regression/SVM, or a decision-tree traversal for Random Forest; do not assume a specific export library not already in `library-docs.md`).

**Inputs / Outputs:**
- Input: exported model parameters.
- Output: on-device `label`/`severity`/`confidence` computed without PC involvement.

**Verification (hardware-dependent):**
- Run the same recorded session through both the PC-side (Feature 27) and on-device inference paths; confirm outputs match within tolerance.

---

### 59 Full Real-Time On-Device Acquisition + Processing Timing Validation

**Goal:** Confirm the ported on-device pipeline (Features 57–58, whichever were completed) meets the real-time timing budget from `architecture.md` → Performance & Timing Constraints.

**Implementation:**
- Instrument the ESP32 firmware to measure per-stage timing (acquisition, filtering, windowing, inference) and report over serial.

**Inputs / Outputs:**
- Output: timing report showing measured latency per stage vs. the documented budget.

**Verification (hardware-dependent, final real-time test):**
- Confirm measured end-to-end on-device latency stays within the sampling interval budget (~10 ms per sample, and the windowing-cadence budget for windowed stages) without dropped samples over an extended run (e.g., 5+ minutes continuous).

---

## Phase N — Final Integration

### 60 Complete End-to-End System Verification

**Goal:** Verify the complete Detect → Analyze → Decide → Mitigate → Measure → Adapt loop operates correctly end-to-end, on real recorded hardware data, through the full production module chain, with no bypassed interfaces.

**Implementation:**
- No new implementation — this is a verification-only feature that exercises every prior feature's production code path together: `firmware` (or a recorded session) → `signal_processing` → `ml`/`estimation` → `controller` → `simulation` → `validation`/`visualization`.
- Run `scripts/run_closed_loop_simulation.py` in recorded-data mode (Feature 43) end-to-end, followed by the full validation suite (Features 44–51) and visualization suite (Features 52–55) against that same session.

**Inputs / Outputs:**
- Input: one or more full recorded ESP32/MPU6050 sessions containing tremor-like and voluntary-movement segments.
- Output: a complete set of validation reports and visualizations for that session, demonstrating every documented success criterion in `architecture.md`/`project-overview.md`.

**Verification (final real-time / end-to-end test):**
- Confirm all of the following simultaneously: acquisition ran without persistent failure (Success Criterion 1), signal processing produced interpretable features (Criterion 2), detection distinguished tremor from voluntary movement on held-out subject-level data (Criterion 3), frequency estimation met its accuracy target once set (Criterion 4), the adaptive controller responded to changing tremor characteristics (Criterion 6), the adaptive strategy used less exposure than fixed mitigation for equivalent suppression (Criterion 7), the full loop ran without interfaces being bypassed (Criterion 8), and robustness tests (Feature 51) passed (Criterion 9). Only when every referenced feature's own verification has passed is the system considered complete — no criterion is asserted based on code existing alone.

---

## Feature Count

| Phase | Features | Cumulative Total |
|---|---|---:|
| Phase 1 — Foundation | 4 | 4 |
| Phase 2 — MPU6050 Acquisition | 5 | 9 |
| Phase 3 — Signal Processing | 12 | 21 |
| Phase 4 — Tremor Detection and ML | 6 | 27 |
| Phase 5 — Frequency, Amplitude, and Phase Estimation | 4 | 31 |
| Phase 6 — Stimulation Simulation | 4 | 35 |
| Phase 7 — Adaptive Controller | 5 | 40 |
| Phase 8 — Closed-Loop Integration | 3 | 43 |
| Phase 9 — Validation and Robustness | 8 | 51 |
| Phase 10 — Visualization and Monitoring | 4 | 55 |
| Phase 11 — ESP32 Real-Time Deployment | 4 | 59 |
| Phase N — Final Integration | 1 | 60 |
| **Total** | **60** | **60** |

---

## Build Order Rules

- Number features sequentially as 01–60 across all phases, in the exact order listed above.
- Every feature has a distinct, testable outcome — no feature is verified by "the code exists and compiles" alone.
- Dependencies are built before the features that consume them (e.g., filtering, Feature 14, before windowing's real use in Feature 16's tests that assume a filtered signal; PSD, Feature 17, before spectral features, Feature 18).
- No feature references a module, interface, file, model, dataset, or configuration not introduced by an earlier feature or already defined in `architecture.md`/`code-standards.md`/`library-docs.md`.
- Each feature is independently verifiable before the next dependent feature begins.
- Features are kept as small vertical slices; no phase contains untestable, purely infrastructural work with no observable output.
- Recorded sensor data or synthetic signals are used for every offline/ML/simulation/controller test; real hardware is required only for Phase 2 and Phase 11 features explicitly marked hardware-dependent.
- No feature requires physical stimulation hardware; the stimulation subsystem (Phase 6) remains simulation-only throughout, including in Phase 9 validation and Phase 11 deployment.
- ML (Phase 4) is not built before the signal-processing and feature-generation pipeline (Phase 3) is verified.
- The adaptive controller (Phase 7) is not built before its detection (Phase 4), analysis/estimation (Phase 5, frequency+amplitude only), and simulation interface (Phase 6) are independently verified.
- Phase-aware control is never built on top of unvalidated phase estimation — Feature 31 (phase estimation) and Feature 50 (phase accuracy validation) remain separate from and are not prerequisites for Phase 7's controller, which operates with `phase=None`.
- Experimentation/validation code (Phase 9) is kept separate from reusable production pipeline code (Phases 3–7), calling the same production functions rather than forked copies.
- No feature duplicates functionality already implemented in an earlier feature (e.g., Feature 15 completes rather than reimplements Feature 12's strongest-axis stub).
- No files, functions, APIs, dependencies, or infrastructure are invented beyond what is defined in `project-overview.md`, `architecture.md`, `code-standards.md`, and `library-docs.md`; where an exact file/module name is not defined by the architecture, it is explicitly marked TBD in this plan rather than invented.
- Implementation details marked TBD in the source context files (filter order, window length, axis strategy, ML/controller thresholds, phase-estimation method, simulation timestep/latency, MATLAB-vs-Python simulation path, on-device export mechanism) remain TBD in this plan and are not silently assigned final values.
- Every feature states its verification method explicitly, and distinguishes hardware-dependent tests, offline signal-processing tests, ML evaluation, simulation tests, closed-loop tests, and final real-time tests where applicable.
- Every feature identifies its produced artifacts (source modules, datasets, model files, plots, metrics/reports, logs, configuration, simulation results) where applicable.
- Architecture boundaries from `architecture.md` and `code-standards.md` are preserved throughout: firmware never contains signal-processing/ML/controller logic; the controller never accesses raw sensor data, signal-processing internals, or ML internals directly; the stimulation subsystem never gains a physical-actuator code path; validation never mutates production/runtime state; visualization never becomes a pipeline dependency.
- The plan concludes (Phase N) with a working, fully-connected Detect → Analyze → Decide → Mitigate → Measure → Adapt pipeline verified end-to-end on real recorded hardware data — not merely a collection of independently-passing but disconnected module tests.
