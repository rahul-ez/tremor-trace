# Code Standards

This file defines the engineering conventions an AI coding agent must follow across every layer of the tremor-detection and adaptive-stimulation system, so the codebase reads as if written by a single consistent senior engineer across sessions.

---

## Engineering Mindset

- Always read `project-overview.md` and `architecture.md` before modifying any code; do not infer architecture from file names alone.
- Understand the existing stage of the pipeline (`firmware → signal_processing → ml/estimation → controller → simulation → validation/visualization`) before adding new processing — new logic belongs in the module that already owns that responsibility.
- Keep changes scoped to the requested task; do not refactor unrelated modules "while you're in there."
- Prefer simple, testable, explicit implementations over clever abstractions, metaprogramming, or premature generalization.
- Never duplicate signal-processing or controller logic in a second location (e.g., re-implementing filtering inside `ml/` or re-implementing thresholding inside `simulation/`).
- Verify assumptions (sampling rate, units, array shape, config values) against `config/system_config.yaml` and existing code before changing them — do not guess.
- Handle failures explicitly per the Error Handling section; never let an error pass silently.
- Preserve existing data formats, function signatures, and module interfaces unless the task explicitly requires changing them; if you must change an interface, update every caller in the same change.
- Do not silently convert an experimental/TBD value (see Open Questions in `architecture.md`) into a permanent hardcoded constant — keep it configurable and marked as tunable.
- Validate changes against recorded sensor data or existing tests whenever possible, rather than reasoning about correctness from code alone.

---

## Python

- Always target Python 3.10+ syntax and standard library features; do not use deprecated APIs.
- Always add type hints to function signatures (parameters and return types); do not add type hints to every local variable.
- Prefer `numpy.typing.NDArray[np.float64]` (or similarly explicit array types) over bare `np.ndarray` in signal-processing and simulation code.
- Never use bare `except:`; always catch specific exception types.
- Always use `logging` (never `print`) for anything other than a one-off ad hoc debugging script.
- Never rely on hidden module-level mutable global state for signal, feature, or controller values; pass state explicitly as function arguments and return values.
- Always separate processing logic (`signal_processing/`, `ml/`, `controller/`, `simulation/`) from experiment/plotting code (`validation/`, `visualization/`, `scripts/`) — a processing module must never import from `visualization/` or `scripts/`.
- Always set an explicit random seed (`np.random.default_rng(seed)`, not the legacy global `np.random.seed`) in any code path involving randomness (train/test split, synthetic tremor generation, ODE noise injection) so results are reproducible.
- Prefer `pathlib.Path` over string concatenation for all file paths.
- Always use pandas for tabular/dataset-level operations (feature tables, dataset splits) and NumPy for array/signal-level operations; do not mix them inside a single tight processing loop.
- Imports must be ordered: standard library → third-party → local project imports, each group separated by a blank line, alphabetized within each group.
- Prefer absolute imports (`from tremor_system.signal_processing import filtering`) over relative imports (`from . import filtering`) for anything outside a package's own `__init__.py`.
- Always document array shape and units in the docstring of any function that accepts or returns a NumPy array.

**Example — function signature and array typing:**
```python
import numpy as np
from numpy.typing import NDArray


def bandpass_filter(
    signal: NDArray[np.float64],
    sample_rate_hz: float,
    band_hz: tuple[float, float],
    order: int,
) -> NDArray[np.float64]:
    """Apply a zero-phase Butterworth band-pass filter.

    Args:
        signal: shape (n_samples,), units: g (accel) or deg/s (gyro).
        sample_rate_hz: sampling rate of `signal`, e.g. 100.0.
        band_hz: (low_hz, high_hz) pass band, e.g. (4.0, 12.0).
        order: filter order (see config/system_config.yaml: signal.filter_order).

    Returns:
        Filtered signal, same shape and units as `signal`.
    """
    ...
```

**Example — reproducible random seed:**
```python
rng = np.random.default_rng(seed=42)  # never np.random.seed(42)
noise = rng.normal(0.0, 0.01, size=n_samples)
```

**Example — explicit exception handling with logging:**
```python
import logging

logger = logging.getLogger(__name__)

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    logger.error("Model artifact not found at %s", model_path)
    raise
```

---

## ESP32 / C++ Firmware

- Always separate `setup()` (one-time initialization: I²C bus, MPU6050 register configuration, serial baud rate) from `loop()`/timer-driven runtime logic; never re-initialize the sensor inside the acquisition loop.
- Always access the MPU6050 through the dedicated driver module (`mpu6050_driver.cpp/.h`); never issue raw `Wire.write`/`Wire.read` calls from `main.cpp`.
- Always drive the 100 Hz acquisition using a hardware timer or a `micros()`-based scheduler with drift correction; never use a plain `delay(10)` in the acquisition loop, since `delay()` drifts and blocks.
- Always timestamp each sample with `micros()` at the moment of read, not at the moment of transmission.
- Prefer fixed-size stack buffers (e.g., `uint8_t buf[14]`) for the raw I²C read block; never use dynamic allocation (`new`/`malloc`) in the acquisition path.
- Never perform blocking operations (e.g., `Serial.print` with implicit flush-wait, long I²C retries) inside the timed acquisition loop; keep the timer-critical path minimal and defer non-critical work.
- Always emit one CSV line per sample in the exact format `timestamp_us,ax,ay,az,gx,gy,gz\n`; never change field order or add fields without updating the Data Contracts section and every downstream parser.
- Always check the I²C transaction return status; on failure, skip the sample and emit a distinct error marker line rather than transmitting stale or zeroed data.
- Always define constants (I²C pins, clock speed, sample rate, DLPF register value, accel/gyro range) as named `constexpr` values at the top of the relevant file; never use unexplained magic numbers inline.
- Keep memory usage minimal and predictable: no per-sample heap allocation, no unbounded buffers, no String concatenation in the acquisition path (use `snprintf` into a fixed `char` buffer instead).
- Keep all hardware-specific code (register addresses, I²C timing, DLPF configuration) inside `firmware/`; never let signal-processing, ML, or controller logic reference ESP32-specific APIs.

**1. Sensor initialization:**
```cpp
void setupMPU6050() {
  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(I2C_CLOCK_HZ);

  writeRegister(MPU6050_ADDR, REG_PWR_MGMT_1, 0x00);
  writeRegister(MPU6050_ADDR, REG_CONFIG, DLPF_CONFIG);
  writeRegister(MPU6050_ADDR, REG_ACCEL_CONFIG, ACCEL_RANGE_2G);
  writeRegister(MPU6050_ADDR, REG_GYRO_CONFIG, GYRO_RANGE_250DPS);
}
```

**2. Timed acquisition (hardware timer callback):**
```cpp
hw_timer_t* samplingTimer = nullptr;

void IRAM_ATTR onSampleTimer() {
  sampleReadyFlag = true;  // set flag, defer actual work to loop()
}

void setupSamplingTimer() {
  samplingTimer = timerBegin(0, 80, true);            // 1 MHz tick
  timerAttachInterrupt(samplingTimer, &onSampleTimer, true);
  timerAlarmWrite(samplingTimer, SAMPLE_INTERVAL_US, true);  // 10000 us = 100 Hz
  timerAlarmEnable(samplingTimer);
}
```

**3. Reading sensor data:**
```cpp
bool readMPU6050Raw(RawSample& out) {
  uint8_t buf[14];
  if (!i2cReadBlock(MPU6050_ADDR, REG_ACCEL_XOUT_H, buf, sizeof(buf))) {
    return false;
  }
  out.timestamp_us = micros();
  out.ax = static_cast<int16_t>((buf[0] << 8) | buf[1]);
  out.ay = static_cast<int16_t>((buf[2] << 8) | buf[3]);
  out.az = static_cast<int16_t>((buf[4] << 8) | buf[5]);
  out.gx = static_cast<int16_t>((buf[8] << 8) | buf[9]);
  out.gy = static_cast<int16_t>((buf[10] << 8) | buf[11]);
  out.gz = static_cast<int16_t>((buf[12] << 8) | buf[13]);
  return true;
}
```

**4. Serial CSV output:**
```cpp
void writeSampleCsv(const RawSample& s) {
  char line[96];
  snprintf(line, sizeof(line), "%lu,%d,%d,%d,%d,%d,%d\n",
           s.timestamp_us, s.ax, s.ay, s.az, s.gx, s.gy, s.gz);
  Serial.print(line);
}
```

**5. Handling sensor/I²C failure:**
```cpp
void acquireLoop() {
  if (!sampleReadyFlag) return;
  sampleReadyFlag = false;

  RawSample sample;
  if (!readMPU6050Raw(sample)) {
    Serial.println("ERR,I2C_READ_FAILED");  // distinct, parseable error marker
    return;  // skip this cycle, do not transmit stale/zero data
  }
  writeSampleCsv(sample);
}
```

---

## Signal Processing

- Always assume the nominal sampling rate is read from `config/system_config.yaml: sensor.sample_rate_hz` (default 100 Hz); never hardcode `100` or `0.01` inline — use the config value or a derived variable.
- Signal units: accelerometer values must be in **g** and gyroscope values in **°/s** after calibration; raw ADC/LSB values must be converted in `calibration.py` and never passed downstream unconverted.
- Array shape convention: a single-channel time series is `shape (n_samples,)`; a multi-axis signal is `shape (n_samples, n_axes)`; a batch of windows is `shape (n_windows, window_samples, n_axes)`. Never silently transpose — validate shape with an assertion at function entry.
- Axis convention: axis order is always `[x, y, z]` for both accel and gyro, matching the firmware CSV field order; never reorder axes implicitly inside a processing function.
- Always implement band-pass filtering with `scipy.signal.butter` + `scipy.signal.filtfilt` (zero-phase) for offline/training-time processing; if a causal (`scipy.signal.lfilter`) version is needed for real-time/on-device use, implement it as a clearly separate function and document the phase-delay difference in its docstring.
- Always read filter band edges and order from config (`signal.tremor_band_hz`, `signal.voluntary_band_hz`, `signal.filter_order`); never hardcode `4, 12` or similar literals inside `filtering.py`.
- Windowing must use the configured window length and overlap (`signal.window_length_s`, `signal.window_overlap_pct`); always discard a trailing incomplete window rather than zero-padding it (see Error Handling).
- Always compute PSD with `scipy.signal.welch`, not a raw `np.fft.fft`, unless a function is explicitly named `*_fft_debug` and documented as a non-standard diagnostic path.
- Feature names must exactly match the schema in `architecture.md` → Interfaces & Data Contracts (`tremor_band_power`, `total_power`, `power_ratio`, `dominant_frequency_hz`, `rms_amplitude`, `variance`, `spectral_entropy`, `accel_magnitude`, `gyro_magnitude`); never introduce an alternate name for the same quantity.
- For numerical stability, always guard divisions (e.g., `power_ratio = tremor_power / (total_power + 1e-12)`) to avoid division-by-zero on silent/stationary segments.
- **Causal vs. non-causal**: `filtfilt`-based filtering, and any technique that uses future samples relative to a given point (e.g., centered windows), is non-causal and offline-only. Any such function's docstring must state `"Offline/non-causal — not valid for real-time on-device use without modification."` A causal, real-time-ready implementation must never be labeled or documented as if it were the offline version, and vice versa.

**Example — explicit shape/unit-documented function:**
```python
def compute_welch_psd(
    signal: NDArray[np.float64],
    sample_rate_hz: float,
    nperseg: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute Welch PSD for a single-axis tremor-band signal.

    Args:
        signal: shape (n_samples,), units: g (post band-pass filtering).
        sample_rate_hz: e.g. 100.0.
        nperseg: samples per Welch segment.

    Returns:
        (freqs_hz, psd) both shape (n_freq_bins,).

    Note: offline/non-causal — uses the full window, not valid as a
    sample-by-sample real-time algorithm.
    """
    freqs, psd = scipy.signal.welch(signal, fs=sample_rate_hz, nperseg=nperseg)
    return freqs, psd
```

---

## Machine Learning

- Always apply the exact same preprocessing (calibration → filtering → windowing → feature extraction) and the exact same fitted `scaler.pkl` at inference time as was used at training time; never fit a new scaler during inference.
- Feature ordering passed into a model must exactly match the ordering used during training; always pass features as a named structure (dict or `pandas.Series` with column labels) through `feature_extraction.py`, and convert to a plain array only at the final `model.predict(...)` call site, to avoid silent column-order bugs.
- Model input contract: a single feature vector per window, matching the schema in `architecture.md`. Model output contract: `{label: bool, severity: float, confidence: float}` — no additional or renamed fields without updating `architecture.md` and every consumer.
- Train/validation/test separation and all cross-validation must be subject-level (grouped by `subject_id`), never window-level; always use `sklearn.model_selection.GroupKFold` or an equivalent explicit grouping — never `train_test_split` without a `groups` parameter on windowed tremor data.
- Always set `random_state=` on every stochastic scikit-learn estimator and splitter for reproducibility.
- Model artifacts must be saved with a versioned filename (`model_<name>_<version>.pkl`) and must never overwrite a previous version silently; always save the paired `scaler.pkl` alongside the model version it was fit for.
- Confidence must come from the model's native probability output (`predict_proba`) or an explicitly calibrated wrapper (`CalibratedClassifierCV`); never fabricate a confidence value from a heuristic when the model doesn't natively support probability output.
- Severity estimation logic (whether model-derived or rule-based on tremor-band power/amplitude) must live behind `ml/inference.py`'s return contract; the calling code (controller) must never need to know which method produced it.
- Model comparison (Logistic Regression vs. SVM vs. Random Forest) must report precision, recall, F1, sensitivity, and false-positive rate per model — never select a model using accuracy alone.
- Keep all ML logic inside `ml/`; a controller or simulation module must never import scikit-learn or call `.predict(...)` directly.
- **The ML model must never directly control stimulation.** `ml/inference.py` returns data only; it must have no code path, callback, or side effect that invokes `controller/` or `simulation/` functions.

---

## Adaptive Controller

- The controller's only inputs are the documented ML/estimation output: `{label, severity, confidence, dominant_frequency_hz, amplitude, phase: Optional[float]}`. The controller must never accept a raw signal array, feature vector, or PSD as input.
- The controller's only outputs are a `StimParams` object (see Data Contracts) and an updated internal `controller_state`; it must never return or log a modified copy of the raw ML/estimation input.
- Confidence gating is mandatory and must be checked first: if `confidence < config.ml.confidence_threshold`, the decision must be `mitigate=False` regardless of severity, and this must be logged as a distinct "low-confidence, no action" state (not silently identical to a genuine no-tremor state).
- All emitted stimulation parameters must be clamped to `config.controller.param_bounds.*`; a parameter selection or adaptation step that would exceed a bound must clamp to the bound and log a `bound_clamped=True` event — never silently exceed or silently clamp without logging.
- Hysteresis must use distinct enter/exit thresholds (`config.controller.hysteresis_pct`) for the mitigate/no-mitigate transition; a single shared threshold for both directions is not acceptable, since it permits chatter.
- Adaptation per cycle must never exceed `config.controller.max_delta_per_step` for any parameter; always clamp the requested delta before applying it.
- Decisions must be deterministic given the same input and controller state — no randomness inside `decision_logic.py` or `adaptation.py`. If exploration/randomized behavior is ever required, it must live in a clearly separate, explicitly named experimental module, not in the production decision path.
- Controller state (current parameters, hysteresis band position, adaptation history, last-decision timestamp) must be held in an explicit `ControllerState` object passed in/out of functions; never a module-level mutable global.
- Target suppression comparison (`achieved_suppression_pct` vs. `config.controller.target_suppression_pct`) must be the sole basis for the increase/decrease/maintain decision in `adaptation.py`; do not fold in other side signals (e.g., raw amplitude) without updating the documented decision logic.
- On invalid/missing input (e.g., `dominant_frequency_hz=None`, `SimResult.stability_warning=True`), the controller must default to `maintain` (no parameter change) rather than guessing a new parameter value.
- Keep controller logic independent of signal-processing and ML implementation details: `controller/` must never import from `signal_processing/` or call scikit-learn directly — only the documented dataclass/dict interfaces cross the boundary.
- **The controller must never access raw MPU6050 data directly** — not via a file path, not via a live serial connection, not via any shared buffer.

---

## Stimulation Simulation

- The simulation's only entry point is `simulation.apply(params: StimParams, tremor_state: TremorState) -> SimResult`; no other function signature should be used to invoke a simulation run from outside `simulation/`.
- `StimParams` fields (`amplitude`, `pulse_frequency_hz`, `pulse_width_us`, `duty_cycle`, `on_off_timing`, `phase: Optional[float]`) must exactly match what `controller/parameter_selection.py` and `controller/adaptation.py` emit; do not add simulation-only fields to this shared contract — use a separate internal config for simulation-only tuning.
- Parameter bounds must be validated at the simulation boundary too (defense in depth), even though the controller is expected to have already clamped them; an out-of-bounds `StimParams` must raise an explicit error rather than being silently accepted.
- Simulation timestep (`config.simulation.timestep_s`) must be small enough relative to the fastest relevant time constant in the ODE/state-space model; document the chosen timestep's justification in a code comment where it is set.
- Latency modeling (`config.simulation.latency_ms`) must be applied explicitly in the simulation loop (e.g., a delay buffer), not approximated by simply reporting a latency number without actually delaying the effect.
- The tremor-response model is a development/validation tool; always include a module-level docstring in `stimulation_model.py` stating it is **not a clinically validated physiological model**.
- `SimResult` must always be returned as the full documented structure (`post_mitigation_signal, achieved_suppression_pct, residual_amplitude, latency_ms`), even in degraded/edge-case runs — use `stability_warning: bool` for instability, never an exception, for numerically recoverable instability (reserve exceptions for truly invalid input).
- All ODE solves (`scipy.integrate.solve_ivp`) must be seeded/deterministic when the model includes any stochastic term, via the same `np.random.default_rng(seed)` convention used elsewhere.
- **The stimulation subsystem must remain simulation-only.** No function in `simulation/` may call a GPIO/DAC/hardware driver, and no function outside `simulation/` may bypass `simulation.apply(...)` to reach a hardware output. No code anywhere in this repository may directly trigger physical electrical stimulation.

---

## Validation and Experiments

- Every validation experiment script (`validation/experiments/*.py`) must set an explicit random seed at the top of the script for full-run reproducibility.
- Evaluation involving recorded (non-synthetic) subject data must reuse the same subject-level split logic as `ml/dataset_builder.py` — never re-derive an ad hoc split inside a validation script.
- The three baseline comparisons (`no_mitigation.py`, `fixed_mitigation.py`, `adaptive_mitigation.py`) must each call the same production `controller/` and `simulation/` functions, differing only in configuration/bypass flags passed in — never a forked copy of controller or simulation logic.
- All metrics (tremor-power reduction, residual amplitude, time-to-target, duty cycle, exposure, latency, false-activation rate, controller stability) must be computed in `validation/metrics.py`, not recomputed inline inside individual experiment scripts.
- Experiment configuration (which strategy, which subject/session, which seed) must be passed explicitly as script arguments or a config file, never hardcoded inside `metrics.py` or the production pipeline modules.
- Results must be written to `data/validation/<experiment_id>/report.json`; validation code must never write into `data/raw/`, `data/processed/`, or `data/models/`.
- Validation and experiment code must never mutate the state of a shared/production `ControllerState` or model artifact — always operate on a fresh instance per run.
- Keep experimental/one-off analysis code in `validation/` or `scripts/`; never let `signal_processing/`, `ml/`, `controller/`, or `simulation/` import from `validation/`.

---

## File and Folder Naming

- Folder names: lowercase `snake_case` (e.g., `signal_processing/`, `data/raw/`).
- Python module files: lowercase `snake_case.py` (e.g., `frequency_estimation.py`), never `CamelCase.py` or `kebab-case.py`.
- C++/Arduino files: lowercase `snake_case.cpp` / `.h`, matching the module name (e.g., `mpu6050_driver.cpp`, `mpu6050_driver.h`).
- Test files: `test_<module_name>.py` inside `tests/`, mirroring the module path being tested (e.g., `tests/test_filtering.py` tests `signal_processing/filtering.py`).
- Model artifacts: `model_<algorithm>_<version>.pkl` (e.g., `model_random_forest_v3.pkl`); the paired scaler is `scaler_<version>.pkl` matching the same version tag.
- Dataset files: `<subject_id>/<session_id>/<stage>.{csv,parquet}` (e.g., `data/raw/subj01/sess02/raw_stream.csv`); never embed spaces or timestamps-as-filenames in place of the structured `subject_id/session_id` hierarchy.
- Experiment/result directories: `data/validation/<experiment_type>_<date_or_run_id>/` (e.g., `data/validation/adaptive_mitigation_20260817/`).

**Correct:** `signal_processing/spectral_analysis.py`, `tests/test_spectral_analysis.py`, `model_svm_v2.pkl`, `data/raw/subj03/sess01/raw_stream.csv`
**Incorrect:** `SignalProcessing/SpectralAnalysis.py`, `spectralAnalysisTest.py`, `svm-model-final-FINAL.pkl`, `data/raw/John's Session (Aug 17).csv`

---

## Module and Component Structure

Every Python module must follow this order:

1. Imports (stdlib → third-party → local)
2. Constants/configuration loading
3. Types/data structures (dataclasses, TypedDicts)
4. Helper functions
5. Main implementation (the module's primary public function(s))
6. Entry point (`if __name__ == "__main__":`, only in scripts, not in library modules)

**Python processing module template:**
```python
# 1. Imports
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tremor_system.config import load_config

# 2. Constants/configuration
logger = logging.getLogger(__name__)
CONFIG = load_config()

# 3. Types/data structures
from dataclasses import dataclass

@dataclass
class WindowResult:
    tremor_band_power: float
    dominant_frequency_hz: float

# 4. Helper functions
def _validate_shape(signal: NDArray[np.float64]) -> None:
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {signal.shape}")

# 5. Main implementation
def analyze_window(signal: NDArray[np.float64], sample_rate_hz: float) -> WindowResult:
    _validate_shape(signal)
    ...
    return WindowResult(tremor_band_power=..., dominant_frequency_hz=...)

# 6. Entry point (scripts only — omit in library modules)
if __name__ == "__main__":
    ...
```

**ESP32 firmware module template (`.cpp`):**
```cpp
// 1. Includes
#include "mpu6050_driver.h"
#include <Wire.h>

// 2. Constants
constexpr uint8_t MPU6050_ADDR = 0x68;
constexpr uint32_t I2C_CLOCK_HZ = 400000;

// 3. Types
struct RawSample { uint32_t timestamp_us; int16_t ax, ay, az, gx, gy, gz; };

// 4. Helper functions
static bool writeRegister(uint8_t addr, uint8_t reg, uint8_t value) { ... }

// 5. Main implementation
void setupMPU6050() { ... }
bool readMPU6050Raw(RawSample& out) { ... }

// 6. Entry point — main.cpp only
// void setup() { ... }  void loop() { ... }
```

**ML module template:**
```python
# 1. Imports
import joblib
import numpy as np
from sklearn.base import ClassifierMixin

# 2. Constants
MODEL_DIR = Path("data/models")

# 3. Types
from dataclasses import dataclass

@dataclass
class InferenceResult:
    label: bool
    severity: float
    confidence: float

# 4. Helpers
def _load_scaler(version: str): ...

# 5. Main implementation
def predict(features: dict[str, float], model_version: str) -> InferenceResult: ...
```

**Controller module template:**
```python
# 1. Imports
from dataclasses import dataclass, field

# 2. Constants (bounds loaded from config)
from tremor_system.config import load_config
CONFIG = load_config()

# 3. Types
@dataclass
class ControllerState:
    current_params: "StimParams"
    hysteresis_active: bool = False

# 4. Helpers
def _clamp(value: float, lo: float, hi: float) -> float: ...

# 5. Main implementation
def decide_and_adapt(ml_output: dict, state: ControllerState) -> tuple["StimParams", ControllerState]: ...
```

**Simulation module template:**
```python
# 1. Imports
from scipy.integrate import solve_ivp

# 2. Constants
from tremor_system.config import load_config
CONFIG = load_config()

# 3. Types
@dataclass
class SimResult:
    achieved_suppression_pct: float
    residual_amplitude: float
    latency_ms: float
    stability_warning: bool = False

# 4. Helpers
def _ode_rhs(t, y, params): ...

# 5. Main implementation
def apply(params: "StimParams", tremor_state) -> SimResult: ...
```

---

## Data Contracts

- **ESP32 CSV format**: `timestamp_us:uint32, ax:int16, ay:int16, az:int16, gx:int16, gy:int16, gz:int16` — raw register values, one line per sample, comma-separated, newline-terminated. Unit conversion to g/°/s happens only in `signal_processing/calibration.py`.
- **Timestamps**: always microseconds (`timestamp_us`), monotonic per session, never wall-clock/epoch time in firmware output.
- **Accelerometer fields**: `ax, ay, az`, units **g** after calibration (raw int16 LSB before calibration), axis order `[x, y, z]`.
- **Gyroscope fields**: `gx, gy, gz`, units **°/s** after calibration (raw int16 LSB before calibration), axis order `[x, y, z]`.
- **Sampling rate**: nominal 100 Hz, sourced from `config.sensor.sample_rate_hz`; actual rate must be measured from `timestamp_us` deltas, never assumed exact.
- **NumPy array shapes**: single-axis signal `(n_samples,)`; multi-axis signal `(n_samples, n_axes)`; batch of windows `(n_windows, window_samples, n_axes)`.
- **Window dimensions**: `window_samples = int(window_length_s * sample_rate_hz)`, `step_samples = int(window_samples * (1 - overlap_pct/100))`.
- **Feature ordering**: `[tremor_band_power, total_power, power_ratio, dominant_frequency_hz, rms_amplitude, variance, spectral_entropy, accel_magnitude, gyro_magnitude]` — this exact order whenever features are converted to a plain array for a model.
- **ML input**: the feature vector above, keyed by name (dict/Series) until the final `.predict()` call. **ML output**: `{label: bool, severity: float, confidence: float, dominant_frequency_hz: float, amplitude: float, phase: float | None}`.
- **Controller input**: the ML output above. **Controller output (`StimParams`)**: `{amplitude: float, pulse_frequency_hz: float, pulse_width_us: float, duty_cycle: float [0.0–1.0], on_off_timing: tuple[float, float], phase: float | None}`.
- **Simulation input**: `StimParams` + `TremorState`. **Simulation output (`SimResult`)**: `{post_mitigation_signal: NDArray, achieved_suppression_pct: float, residual_amplitude: float, latency_ms: float, stability_warning: bool}`.

Every new or modified data contract must be reflected in both this file and `architecture.md` in the same change — never update one without the other.

---

## Configuration and Constants

- Sampling rate, filter cutoffs, filter order, window size, overlap, tremor-band limits, ML thresholds, controller thresholds, hysteresis, stimulation bounds, and simulation parameters all have exactly one source of truth: `config/system_config.yaml`.
- Always load configuration through a single typed loader (e.g., `tremor_system.config.load_config()`); never `yaml.safe_load` the config file directly from inside a processing module.
- Never duplicate the same configurable value across multiple source files — if two modules need the same threshold, both must read it from the same config object, not from two separately hardcoded literals that happen to match today.
- When a config value is experimentally undetermined (TBD in `architecture.md`), it must still exist in `system_config.yaml` with a clearly marked placeholder/default and a comment referencing the relevant Open Question — never omitted or hardcoded elsewhere "temporarily."
- Any change to a shared config value must be made once, in `system_config.yaml`, and never patched around with a local override inside a single module.

---

## Error Handling

- Raise an exception when a precondition is violated that the caller could have prevented (e.g., malformed config, missing model file, invalid array shape) — these are programmer/setup errors, not runtime conditions to route around.
- Return an explicit status/result field (e.g., `stability_warning: bool`, `label: None`) when a condition is a normal, expected runtime outcome (e.g., low-confidence detection, unstable-but-recoverable simulation step, incomplete window) — do not raise for these.
- Always log at `ERROR` level before re-raising, and at `WARNING` level for recoverable/degraded conditions (e.g., `bound_clamped=True`, `stability_warning=True`).
- Sensor data unavailable (I²C failure, dropped serial line): skip the sample, log a distinct marker, and do not fabricate/interpolate a replacement sample.
- ML confidence insufficient: propagate `confidence` through unchanged; the controller — not the ML module — is responsible for deciding `mitigate=False` on low confidence.
- Simulation inputs invalid (out-of-bounds `StimParams`, malformed `TremorState`): raise `ValueError` with the specific invalid field named in the message; do not silently clamp at this layer without logging (bounds should already have been enforced by the controller — this is a defense-in-depth check).
- Never silently ignore an error: no bare `except: pass`, no swallowed exception without at least a `logger.warning(...)` or `logger.error(...)` call.

---

## Logging and Debugging

- Python logging format: `%(asctime)s %(levelname)s %(name)s: %(message)s`, configured once at the application entry point (script `__main__`), never reconfigured inside library modules.
- Severity levels: `DEBUG` for per-sample/per-window internal values during development; `INFO` for pipeline-stage completion (e.g., "loaded model v3", "processed 240 windows"); `WARNING` for recoverable/degraded conditions; `ERROR` for exceptions about to be raised or caught-and-handled failures.
- Firmware serial output has two disjoint line types: machine-readable data lines (`timestamp_us,ax,ay,az,gx,gy,gz`) and error marker lines (`ERR,<REASON>`); never mix a human-readable debug string into the same stream without a distinguishing prefix, since Serial Studio and the PC-side logger parse the stream positionally/by prefix.
- Log timestamps in Python using ISO 8601 via the standard `logging` `asctime` formatter; log timestamps in firmware using `timestamp_us` (`micros()`), never a formatted date string (the ESP32 has no reliable wall clock in current scope).
- Log what is needed for debugging and validation (stage completion, key metrics, error conditions) — never log full raw signal arrays or full feature tables at `INFO` level; use `DEBUG` and gate behind an explicit debug flag if full-array logging is ever needed.
- Debug/diagnostic output (ad hoc `print`, exploratory plots, scratch scripts) must never become a dependency of downstream processing — no production code path may parse or rely on a debug log line's content.
- Machine-readable serial output format must remain stable: any change to the CSV field order, count, or type must be treated as a breaking data-contract change (see Data Contracts) and reflected in `architecture.md`, not made silently.

---

## Imports and Dependencies

- Import ordering: standard library, then third-party, then local project imports, each group alphabetized and separated by a blank line.
- Prefer absolute imports for cross-package references; relative imports (`from . import x`) are acceptable only within a single package's own submodules.
- Dependency direction must follow the pipeline: `firmware → signal_processing → ml/estimation → controller → simulation → validation/visualization`. A module must only import from modules at or before its own stage; it must never import from a later stage (e.g., `signal_processing/` must never import from `controller/`).
- Circular imports are prohibited; if two modules seem to need each other, extract the shared piece into a lower-level shared module (e.g., `types.py` or `config.py`) that both depend on instead.
- Only the packages listed in Approved Dependencies may be imported. Introducing a new dependency requires updating the Approved Dependencies table in the same change, with a stated purpose and required/optional status.
- Do not introduce a web framework, database client, authentication library, or cloud SDK — none is required by this architecture (see `architecture.md`: Stack, `None` for Database/Auth/Payments/Cloud Backend).

---

## Comments

- Comments should explain *why* non-obvious code exists (a physiological assumption, a numerical-stability workaround, a hardware quirk), not restate *what* the code does line-by-line.
- Always document signal-processing assumptions and units at the point they are introduced (e.g., "gyro values already converted to °/s by calibration.py — do not re-scale here").
- Always document experimental/TBD decisions explicitly in a comment referencing the corresponding Open Question in `architecture.md` (e.g., `# TBD: filter order — see architecture.md Open Questions #1`).
- Never use a comment to justify violating an architectural boundary (e.g., `# controller reads raw signal here just this once` is not acceptable — fix the boundary violation instead).
- Do not leave unresolved `TODO` comments in code intended for merge; either resolve the TODO, file it as a tracked Open Question in `architecture.md`, or remove the code path.

---

## Architecture Boundaries

The following boundaries must always be preserved:

```text
ESP32 acquisition
  → signal processing
  → ML
  → frequency/phase estimation
  → controller
  → stimulation simulation
  → validation/measurement
```

- Hardware code (`firmware/`) must not depend on ML or controller code, and must not contain any signal-processing, ML, or controller logic.
- Signal processing (`signal_processing/`) must not depend on the controller or simulation.
- ML (`ml/`) must not directly control stimulation; it returns data only.
- The controller (`controller/`) must not implement signal-processing or ML algorithms, and must not access raw sensor data.
- Simulation (`simulation/`) must not contain controller decision logic (no thresholding, no confidence gating, no adaptation math inside `simulation/`).
- Validation (`validation/`) must not modify runtime/production components or state; it calls production functions read-only, with fresh instances per run.
- Visualization (`visualization/`) must consume defined outputs (files in `data/`, or objects explicitly passed by orchestration scripts) rather than accessing internal module state or calling control functions.
- Shared data structures (`StimParams`, `SimResult`, feature-vector schema) must not import higher-level business logic (e.g., a `StimParams` dataclass file must not import from `controller/decision_logic.py`).
- Circular dependencies are prohibited at every layer.

---

## Approved Dependencies

| Dependency | Purpose | Used By | Required/Optional |
|---|---|---|---|
| Arduino-ESP32 / ESP-IDF | ESP32 build/runtime framework | `firmware/` | Required |
| Wire (I²C library) | I²C bus communication with MPU6050 | `firmware/` | Required |
| Python 3.10+ | Runtime for all non-firmware code | All Python modules | Required |
| NumPy | Array math, signal representation | `signal_processing/`, `simulation/`, `estimation/` | Required |
| SciPy | Filtering (`butter`, `filtfilt`), Welch PSD, ODE solving (`solve_ivp`) | `signal_processing/`, `estimation/`, `simulation/` | Required |
| pandas | Tabular dataset/feature-table handling | `signal_processing/feature_extraction.py`, `ml/`, `validation/` | Required |
| scikit-learn | Classical ML models, metrics, `GroupKFold` splitting | `ml/` | Required |
| joblib | Model/scaler serialization | `ml/train.py`, `ml/inference.py` | Required |
| PyYAML | Parsing `system_config.yaml` | `config/` loader | Required |
| matplotlib | Static plotting for visualization/reports | `visualization/` | Required |
| pytest | Automated testing | `tests/` | Required (dev) |
| MATLAB/Simulink | Alternative stimulation-response modeling environment | `simulation/` (alternative path to Python ODE) | Optional |
| Serial Studio | Live raw-telemetry visualization | External tool, attaches to serial port | Optional |

The agent must not introduce a new dependency without first adding it to this table with a stated purpose and required/optional status, in the same change that introduces its use.

---

## Verification Checklist

Before considering any change complete, verify:

- [ ] Does the change respect the architecture boundaries (dependency direction, no circular imports, correct module ownership)?
- [ ] Did it preserve existing data contracts (CSV format, feature vector schema, `StimParams`/`SimResult` shapes) unless the task explicitly required changing them?
- [ ] Are units (g, °/s, Hz, %, seconds) and array shapes correct and documented at every function boundary touched?
- [ ] Is the change reproducible (explicit random seeds, deterministic config-driven behavior)?
- [ ] Did it introduce duplicated configuration instead of reading from `system_config.yaml`?
- [ ] Did it introduce a circular dependency or a dependency pointing to a later pipeline stage?
- [ ] Does the relevant test/validation code still pass (`pytest`, and applicable `validation/experiments/*.py`)?
- [ ] If the change affects signal processing, was it tested against recorded sensor data, not synthetic data alone?
- [ ] If the change affects the controller, were bounded parameters, hysteresis, adaptation-rate cap, and low-confidence/invalid-input failure cases explicitly checked?
- [ ] If the change affects firmware, was 100 Hz acquisition timing and the exact serial CSV format preserved (or, if intentionally changed, was the Data Contracts section updated in the same change)?
