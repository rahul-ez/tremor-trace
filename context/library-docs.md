# Library Docs

This file documents how **this project** — the ESP32 + MPU6050 tremor detection and adaptive electrical-stimulation-simulation system — uses each third-party library and tool it depends on. It records approved versions/patterns, project-specific configuration, data contracts, and integration constraints. General library documentation is not a substitute for this file; this file captures how each dependency is actually wired into this codebase's pipeline (`firmware → signal_processing → ml/estimation → controller → simulation → validation/visualization`).

**Authority order:**
1. Current official library/tool documentation or MCP-provided documentation
2. Installed project skills/instructions
3. This file
4. General model knowledge

When this file conflicts with general library knowledge, follow this file unless the architecture (`architecture.md`) or current official documentation explicitly requires otherwise.

---

## Before Using Any Library

1. Check `project-overview.md` and `architecture.md` for the intended role of the dependency.
2. Check `AGENTS.md` and any installed skills for project-wide conventions.
3. Check available MCP/tool documentation when applicable.
4. Read this file for project-specific integration rules, configuration, and data contracts.
5. Inspect existing usage in the codebase before introducing a new pattern — search for how the dependency is already called elsewhere in the relevant module.
6. Reuse existing project patterns instead of creating a second integration approach for the same task.
7. Never add a new dependency simply because it is convenient.

If a dependency or usage pattern is marked **TBD**, **optional**, or **experimental** anywhere in this file, the agent must not treat it as a mandatory production dependency — keep it behind configuration and do not silently promote it to a required code path.

---

## Arduino / ESP32 Arduino Core

**Check first:** Official Arduino-ESP32 documentation (espressif/arduino-esp32 repo)

**Used by:** Firmware

**Purpose:** Provides the runtime, board support, and hardware-access APIs (I²C, timers, Serial) that the ESP32 firmware uses to acquire IMU data and stream it over USB serial.

### Timer-driven sampling

```cpp
#include <Arduino.h>

hw_timer_t* samplingTimer = nullptr;
volatile bool sampleReadyFlag = false;

constexpr uint32_t SAMPLE_RATE_HZ = 100;
constexpr uint32_t SAMPLE_INTERVAL_US = 1000000 / SAMPLE_RATE_HZ;  // 10000 us

void IRAM_ATTR onSampleTimer() {
  sampleReadyFlag = true;
}

void setupSamplingTimer() {
  samplingTimer = timerBegin(0, 80, true);   // prescaler 80 → 1 MHz tick on 80 MHz APB clock
  timerAttachInterrupt(samplingTimer, &onSampleTimer, true);
  timerAlarmWrite(samplingTimer, SAMPLE_INTERVAL_US, true);
  timerAlarmEnable(samplingTimer);
}
```

### Serial output

```cpp
void setup() {
  Serial.begin(115200);
  setupMPU6050();
  setupSamplingTimer();
}

void loop() {
  if (!sampleReadyFlag) return;
  sampleReadyFlag = false;

  RawSample sample;
  if (readMPU6050Raw(sample)) {
    char line[96];
    snprintf(line, sizeof(line), "%lu,%d,%d,%d,%d,%d,%d\n",
             sample.timestamp_us, sample.ax, sample.ay, sample.az,
             sample.gx, sample.gy, sample.gz);
    Serial.print(line);
  } else {
    Serial.println("ERR,I2C_READ_FAILED");
  }
}
```

### Project Integration

The Arduino-ESP32 core sits entirely inside `firmware/`. It owns board init, timer-based scheduling, and `Serial` I/O. It must never be linked to or referenced by any Python module — the boundary between firmware and PC-side processing is the USB-serial CSV stream, nothing else.

### Configuration

| Parameter | Value | Source of Truth |
|---|---|---|
| Serial baud rate | 115200 | `firmware/src/serial_protocol.h` constant, mirrored in PC-side serial reader config |
| Sample rate | 100 Hz target | `config/system_config.yaml: sensor.sample_rate_hz` (firmware constant must match) |
| Timer prescaler | 80 (→ 1 MHz tick on 80 MHz APB clock) | `firmware/src/sampling_timer.cpp` |

### Data Contract

| Field | Type | Units | Notes |
|---|---|---|---|
| `timestamp_us` | uint32 | microseconds | from `micros()`, monotonic per session |
| `ax, ay, az` | int16 | raw LSB | converted to g on the PC side |
| `gx, gy, gz` | int16 | raw LSB | converted to °/s on the PC side |

**Rules:**
- Never call `delay()` inside the acquisition timing path; use the hardware timer + flag pattern shown above.
- Never perform blocking `Serial.print` calls inside the timer ISR itself — only set a flag in the ISR, and do the actual `Serial.print` in `loop()`.
- Never change the serial baud rate or CSV field order without updating this file, `architecture.md`, and the PC-side reader together.
- Do not use Arduino `String` objects in the acquisition/serial-output path — use fixed `char` buffers with `snprintf` to avoid heap fragmentation.

---

## Wire (I²C)

**Check first:** Official Arduino `Wire` library documentation

**Used by:** Firmware

**Purpose:** Provides I²C bus communication between the ESP32 and the MPU6050 for register configuration and raw sample reads.

### Bus initialization

```cpp
#include <Wire.h>

constexpr int PIN_SDA = 21;
constexpr int PIN_SCL = 22;
constexpr uint32_t I2C_CLOCK_HZ = 400000;

void setupI2C() {
  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(I2C_CLOCK_HZ);
}
```

### Register read/write

```cpp
constexpr uint8_t MPU6050_ADDR = 0x68;

bool writeRegister(uint8_t addr, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;  // 0 == success
}

bool i2cReadBlock(uint8_t addr, uint8_t startReg, uint8_t* buf, size_t len) {
  Wire.beginTransmission(addr);
  Wire.write(startReg);
  if (Wire.endTransmission(false) != 0) return false;  // repeated start
  size_t received = Wire.requestFrom(addr, static_cast<uint8_t>(len));
  if (received != len) return false;
  for (size_t i = 0; i < len; i++) buf[i] = Wire.read();
  return true;
}
```

### Project Integration

`Wire` usage is fully encapsulated inside `firmware/src/mpu6050_driver.cpp/.h`. No other firmware file (e.g., `main.cpp`) may call `Wire.*` functions directly — all I²C access goes through the driver module's functions (`writeRegister`, `i2cReadBlock`).

### Configuration

| Parameter | Value | Source of Truth |
|---|---|---|
| SDA pin | GPIO 21 | `firmware/src/mpu6050_driver.h` |
| SCL pin | GPIO 22 | `firmware/src/mpu6050_driver.h` |
| I²C clock | 400 kHz | `firmware/src/mpu6050_driver.h`, mirrored in `config/system_config.yaml: sensor.i2c_clock_hz` |
| MPU6050 address | 0x68 | `firmware/src/mpu6050_driver.h` |

### Data Contract

I²C reads return a raw 14-byte block: `ax_h, ax_l, ay_h, ay_l, az_h, az_l, temp_h, temp_l, gx_h, gx_l, gy_h, gy_l, gz_h, gz_l`. Only the accel and gyro bytes are used; temperature bytes are read (to keep the block contiguous) but discarded.

**Rules:**
- Always check the return value of `Wire.endTransmission()` and `Wire.requestFrom()`; never assume a transaction succeeded.
- Always use a repeated start (`Wire.endTransmission(false)`) when writing the register pointer before a `requestFrom` read, to avoid releasing the bus mid-transaction.
- Never change `I2C_CLOCK_HZ` without verifying the MPU6050 datasheet's supported clock range — an incorrect clock configuration is a common source of intermittent read failures.
- Never retry a failed I²C transaction in a blocking loop inside the timed acquisition path; skip the sample and log `ERR,I2C_READ_FAILED` instead (see `code-standards.md` → Error Handling).

---

## Python

**Check first:** Official Python 3.10+ documentation

**Used by:** Signal Processing, ML, Estimation, Controller, Simulation, Validation, Visualization

**Purpose:** Runtime for every PC-side component of the pipeline — from calibration through validation.

### Project entrypoint pattern

```python
# scripts/run_pipeline_baseline.py
import logging

from tremor_system.config import load_config
from tremor_system.signal_processing import calibration, filtering, windowing, spectral_analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    ...


if __name__ == "__main__":
    main()
```

### Project Integration

Python is the runtime for everything outside `firmware/`. Every non-firmware module in `architecture.md`'s folder structure (`signal_processing/`, `ml/`, `estimation/`, `controller/`, `simulation/`, `validation/`, `visualization/`) is a Python package.

### Configuration

| Parameter | Value | Source of Truth |
|---|---|---|
| Python version | 3.10+ | Project-wide requirement (see `code-standards.md`) |
| Logging format | `%(asctime)s %(levelname)s %(name)s: %(message)s` | Configured once at each script entry point |

### Data Contract

N/A — Python itself has no data contract; contracts are defined per-library/module below.

**Rules:**
- Always add type hints to function signatures across all Python modules.
- Never configure `logging.basicConfig` inside a library module — only inside a script's `if __name__ == "__main__":` block.
- Follow the import ordering and module-structure template defined in `code-standards.md`.

---

## NumPy

**Check first:** Official NumPy documentation

**Used by:** Signal Processing, Estimation, Simulation

**Purpose:** Represents raw and processed IMU signals as arrays and performs vectorized numerical operations (calibration offsets, magnitude computation, RMS).

### Array shape and axis convention

```python
import numpy as np
from numpy.typing import NDArray


def compute_magnitude(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute 3-axis magnitude.

    Args:
        signal: shape (n_samples, 3), axis order [x, y, z], units: g or deg/s.

    Returns:
        shape (n_samples,), same units as input.
    """
    return np.linalg.norm(signal, axis=1)
```

### RMS / amplitude computation

```python
def compute_rms(signal: NDArray[np.float64]) -> float:
    """Compute RMS amplitude of a single-axis tremor-band signal.

    Args:
        signal: shape (n_samples,), units: g, post band-pass filtering.

    Returns:
        RMS amplitude, units: g.
    """
    return float(np.sqrt(np.mean(np.square(signal))))
```

### Project Integration

NumPy arrays are the data representation from `signal_processing/calibration.py` through `estimation/` and `simulation/`. `pandas` DataFrames are used only at the dataset/feature-table level (see pandas section) — never inside the tight per-sample/per-window numerical loops.

### Configuration

| Parameter | Value | Source of Truth |
|---|---|---|
| Array dtype | `np.float64` for all signal arrays | Project convention (see `code-standards.md`) |
| Random generator | `np.random.default_rng(seed)` | Never the legacy global `np.random.seed` |

### Data Contract

| Representation | Shape | Notes |
|---|---|---|
| Single-axis signal | `(n_samples,)` | e.g., filtered tremor-band signal for one axis |
| Multi-axis signal | `(n_samples, n_axes)` | axis order `[x, y, z]`, `n_axes` = 3 |
| Batch of windows | `(n_windows, window_samples, n_axes)` | produced by `windowing.py` |

**Rules:**
- Never silently transpose or reshape an array to "make it fit" — validate shape with an explicit assertion/error at function entry instead.
- Always document units in the docstring of any function accepting/returning a NumPy array (g, °/s, Hz, etc.) — unit confusion between raw and calibrated values is a known project risk.
- Always use `np.random.default_rng(seed)` for any stochastic operation (synthetic tremor generation, ODE noise) to keep results reproducible.
- Never perform vectorized NumPy operations across a batch that mixes samples from different subjects/sessions without explicit grouping — this is a common source of accidental subject-level data leakage further downstream in `ml/`.

---

## SciPy

**Check first:** Official SciPy documentation (`scipy.signal`, `scipy.integrate`)

**Used by:** Signal Processing, Estimation, Simulation

**Purpose:** Provides band-pass filtering (`scipy.signal.butter`/`filtfilt`), spectral analysis (`scipy.signal.welch`), and ODE solving (`scipy.integrate.solve_ivp`) for the simulated stimulation-response model.

### Band-pass filtering (offline/non-causal)

```python
import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt


def bandpass_filter(
    signal: NDArray[np.float64],
    sample_rate_hz: float,
    band_hz: tuple[float, float],
    order: int,
) -> NDArray[np.float64]:
    """Zero-phase Butterworth band-pass filter.

    Offline/non-causal — uses filtfilt (forward-backward filtering).
    Not valid as-is for real-time, sample-by-sample on-device processing.

    Args:
        signal: shape (n_samples,), units: g (post calibration).
        sample_rate_hz: e.g. 100.0.
        band_hz: (low_hz, high_hz), e.g. (4.0, 12.0) for the tremor band.
        order: filter order, from config.signal.filter_order.

    Returns:
        Filtered signal, shape (n_samples,), same units as input.
    """
    nyquist = sample_rate_hz / 2.0
    low, high = band_hz[0] / nyquist, band_hz[1] / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)
```

### Welch PSD

```python
from scipy.signal import welch


def compute_welch_psd(
    signal: NDArray[np.float64],
    sample_rate_hz: float,
    nperseg: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute Welch PSD for a single-axis, band-pass-filtered signal.

    Args:
        signal: shape (n_samples,), units: g.
        sample_rate_hz: e.g. 100.0.
        nperseg: samples per Welch segment (from config.signal.window_length_s * sample_rate_hz).

    Returns:
        (freqs_hz, psd), both shape (n_freq_bins,).
    """
    freqs, psd = welch(signal, fs=sample_rate_hz, nperseg=nperseg)
    return freqs, psd
```

### ODE-based stimulation-response simulation

```python
from scipy.integrate import solve_ivp


def simulate_tremor_response(params: StimParams, tremor_state: dict) -> tuple[NDArray[np.float64], bool]:
    """Integrate the simulated tremor/stimulation-response ODE.

    Not a clinically validated physiological model — development/validation only.

    Args:
        params: Shared stimulation parameters; amplitude controls simulated damping.
        tremor_state: Dict with ``y0`` shape (2,), ``duration_s``,
            ``timestep_s``, ``signal`` shape (n_samples,), and
            ``sample_rate_hz``.

    Returns:
        Post-mitigation signal shape (n_samples,) and a stability-warning flag.
    """
    def rhs(t, y):
        ...
    time_s = np.arange(tremor_state["signal"].size) / tremor_state["sample_rate_hz"]
    result = solve_ivp(
        rhs,
        (0.0, tremor_state["duration_s"]),
        tremor_state["y0"],
        t_eval=time_s,
        max_step=tremor_state["timestep_s"],
    )
    return result.y[0], not result.success
```

### Project Integration

SciPy filtering and spectral functions live inside `signal_processing/filtering.py` and `signal_processing/spectral_analysis.py` only. SciPy's `solve_ivp` lives inside `simulation/stimulation_model.py` only. No other module (ML, controller) should call `scipy.signal` or `scipy.integrate` directly.

### Configuration

| Parameter | Value | Source of Truth |
|---|---|---|
| Tremor band | 4–12 Hz | `config.signal.tremor_band_hz` |
| Voluntary band | 0.5–4 Hz | `config.signal.voluntary_band_hz` |
| Filter order | TBD (experimental) | `config.signal.filter_order` |
| Window length | 1–2 s (experimental) | `config.signal.window_length_s` |
| Window overlap | 50% | `config.signal.window_overlap_pct` |
| Simulation timestep | 0.001 s (v1 provisional) | `config.simulation.timestep_s` |
| Simulation latency | 50 ms (v1 provisional) | `config.simulation.latency_ms` |

### Data Contract

| Function | Input | Output |
|---|---|---|
| `bandpass_filter` | `(n_samples,)` float64, units g/°/s | `(n_samples,)` float64, same units |
| `compute_welch_psd` | `(n_samples,)` float64 | `freqs_hz (n_freq_bins,)`, `psd (n_freq_bins,)` |
| `simulate_tremor_response` | `y0 (n_state_vars,)`, `t_span`, `params` | `OdeResult.t`, `OdeResult.y` |

**Rules:**
- Always use `filtfilt` (zero-phase) for offline/training-time filtering; if a causal, real-time-capable filter is needed, implement it as a separately named function (e.g., `bandpass_filter_causal`) using `lfilter`, and document the phase-delay difference explicitly — never let an offline `filtfilt`-based function be silently reused as if it were real-time-safe.
- Always use `scipy.signal.welch`, not `np.fft.fft`, for the project's standard PSD computation; a raw-FFT function must be named distinctly (e.g., `compute_fft_debug`) if it exists at all.
- Always read filter band edges and order from config — never hardcode `4, 12` or a literal order inline in `filtering.py`.
- Always label `simulate_tremor_response` and any function in `simulation/` with a docstring stating it is not a clinically validated physiological model, to prevent simulation assumptions from being mistaken for real physiological behavior downstream (e.g., in a validation report).
- Never call `solve_ivp` without an explicit `max_step` (or equivalent) tied to `config.simulation.timestep_s` — an unconstrained adaptive step can silently produce a coarser-than-intended simulation.

---

## pandas

**Check first:** Official pandas documentation

**Used by:** Signal Processing (feature-table assembly), ML, Validation

**Purpose:** Loads and manipulates tabular datasets — feature tables, dataset splits, and validation result tables. Not used for real-time/per-sample signal processing.

### Feature table assembly

```python
import pandas as pd


def build_feature_table(feature_records: list[dict]) -> pd.DataFrame:
    """Assemble a feature table from per-window feature dicts.

    Each record must include: subject_id, session_id, window_id, and the
    documented feature columns (see architecture.md Interfaces & Data Contracts).
    """
    df = pd.DataFrame.from_records(feature_records)
    required_cols = {
        "subject_id", "session_id", "window_id",
        "tremor_band_power", "total_power", "power_ratio",
        "dominant_frequency_hz", "rms_amplitude", "variance",
        "spectral_entropy", "accel_magnitude", "gyro_magnitude",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Feature table missing required columns: {missing}")
    return df
```

### Subject-level split preparation

```python
def load_dataset(features_path: str) -> pd.DataFrame:
    """Load a features.csv file, preserving subject_id/session_id for grouping."""
    return pd.read_csv(features_path, dtype={"subject_id": str, "session_id": str})
```

### Project Integration

pandas is used at the dataset/feature-table level in `signal_processing/feature_extraction.py` (output only), `ml/dataset_builder.py`, `ml/train.py`, `ml/evaluate.py`, and `validation/metrics.py`. It must never appear inside `signal_processing/filtering.py`, `signal_processing/windowing.py`, or `signal_processing/spectral_analysis.py`, which operate on NumPy arrays.

### Configuration

| Parameter | Value | Source of Truth |
|---|---|---|
| Feature CSV path pattern | `data/features/<subject_id>/<session_id>/features.csv` | `architecture.md` → Storage |
| Required grouping columns | `subject_id`, `session_id` | `code-standards.md` → Data Contracts |

### Data Contract

Feature table columns: `subject_id (str), session_id (str), window_id (int), tremor_band_power (float), total_power (float), power_ratio (float), dominant_frequency_hz (float), rms_amplitude (float), variance (float), spectral_entropy (float), accel_magnitude (float), gyro_magnitude (float)`.

**Rules:**
- Never use pandas inside the tight per-sample/per-window signal-processing loop (`filtering.py`, `windowing.py`, `spectral_analysis.py`) — those stay in NumPy; pandas is for the assembled table only, since DataFrame overhead is not appropriate for latency-sensitive processing.
- Always preserve `subject_id` and `session_id` as string-typed columns through every pandas transformation — never let them be silently coerced to int/float, which can corrupt grouping in `GroupKFold`.
- Always validate that required feature columns are present (see `build_feature_table` example) before handing a DataFrame to `ml/train.py` — never assume column presence implicitly.
- Never fit a `sklearn` scaler or model directly on a pandas DataFrame without first confirming feature column order matches the documented schema (see Data Contracts in `code-standards.md`).

---

## scikit-learn

**Check first:** Official scikit-learn documentation

**Used by:** ML

**Purpose:** Provides classical ML models (Logistic Regression, SVM, Random Forest), preprocessing (scaling), and evaluation utilities for tremor/no-tremor classification and severity/confidence estimation.

### Subject-level splitting

```python
from sklearn.model_selection import GroupKFold


def get_subject_level_folds(X, y, groups, n_splits: int = 5):
    """Yield subject-level train/test indices — never window-level.

    Args:
        X: feature array, shape (n_windows, n_features).
        y: labels, shape (n_windows,).
        groups: subject_id per window, shape (n_windows,).
    """
    gkf = GroupKFold(n_splits=n_splits)
    yield from gkf.split(X, y, groups=groups)
```

### Model training and comparison

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CANDIDATE_MODELS = {
    "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
    "svm": SVC(probability=True, random_state=42),
    "random_forest": RandomForestClassifier(random_state=42, n_estimators=100),
}


def train_and_scale(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit ONLY on training data
    fitted_models = {
        name: model.fit(X_train_scaled, y_train)
        for name, model in CANDIDATE_MODELS.items()
    }
    return fitted_models, scaler
```

### Inference with confidence

```python
import joblib
from dataclasses import dataclass


@dataclass
class InferenceResult:
    label: bool
    confidence: float


def predict(features: dict[str, float], model_path: str, scaler_path: str) -> InferenceResult:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    feature_order = [
        "tremor_band_power", "total_power", "power_ratio", "dominant_frequency_hz",
        "rms_amplitude", "variance", "spectral_entropy", "accel_magnitude", "gyro_magnitude",
    ]
    x = [[features[name] for name in feature_order]]
    x_scaled = scaler.transform(x)  # never scaler.fit_transform at inference time

    label = bool(model.predict(x_scaled)[0])
    confidence = float(model.predict_proba(x_scaled)[0].max())
    return InferenceResult(label=label, confidence=confidence)
```

### Project Integration

scikit-learn usage is fully contained within `ml/`. `controller/` and `simulation/` must never import scikit-learn or call `.predict()`/`.fit()` directly — they consume only the `InferenceResult`-style dict/dataclass returned by `ml/inference.py`.

### Configuration

| Parameter | Value | Source of Truth |
|---|---|---|
| Candidate models | Logistic Regression, SVM, Random Forest | `architecture.md` → ML Architecture |
| Random seed | 42 (project convention; confirm/centralize via config) | `config.ml.random_seed` |
| Confidence threshold | TBD (experimental) | `config.ml.confidence_threshold` |
| Split strategy | `GroupKFold` on `subject_id` | `architecture.md` → Invariants |

### Data Contract

Model input: ordered feature array matching `feature_order` above. Model output: `InferenceResult{label: bool, confidence: float}`; severity is computed separately (rule-based or model-derived) and combined into the full ML output contract documented in `architecture.md`.

**Rules:**
- Always fit the `StandardScaler` (or equivalent) only on training data; never call `fit_transform` on validation/test data — always `transform` only.
- Always split by `subject_id` using `GroupKFold` or equivalent; never use a plain random `train_test_split` on windowed data, since overlapping windows from the same subject will leak across the split.
- Always use the exact same fitted scaler (loaded from `scaler_<version>.pkl`) at inference time as was used at training time — never re-fit a scaler during inference.
- Always preserve feature ordering exactly as defined in the Data Contract; never pass a differently-ordered array into `model.predict()`.
- Never select a model using accuracy alone — report precision, recall, F1, sensitivity, and false-positive rate as required by `architecture.md` → ML Architecture.
- Never call `model.predict()` or any scikit-learn function from `controller/` or `simulation/` — ML inference is fully encapsulated in `ml/inference.py`.

---

## MATLAB/Simulink OR Python-based ODE/state-space simulation

**Check first:** Official MATLAB/Simulink documentation (if used) or `scipy.integrate` documentation (Python path) — use this file for which path is active in the current implementation.

**Used by:** Simulation

**Purpose:** Models the simulated relationship between selected stimulation parameters and tremor suppression, for controller development and validation — not a clinically validated physiological model.

### Python ODE/state-space path (primary, if MATLAB/Simulink not in use)

```python
from dataclasses import dataclass

from scipy.integrate import solve_ivp


@dataclass
class StimParams:
    amplitude: float
    pulse_frequency_hz: float
    pulse_width_us: float
    duty_cycle: float
    on_off_timing: tuple[float, float]
    phase: float | None = None


@dataclass
class SimResult:
    achieved_suppression_pct: float
    residual_amplitude: float
    latency_ms: float
    stability_warning: bool = False


def apply(params: StimParams, tremor_state: dict) -> SimResult:
    """Simulate stimulation response. Simulation-only — no physical actuation."""
    def rhs(t, y):
        # y: [tremor_amplitude, ...simulation state...]
        ...
    sol = solve_ivp(rhs, (0, tremor_state["duration_s"]), tremor_state["y0"],
                     max_step=tremor_state["timestep_s"])
    ...
    return SimResult(achieved_suppression_pct=..., residual_amplitude=..., latency_ms=...)
```

### MATLAB/Simulink path (alternative, if selected instead of the Python path)

Integration pattern: **TBD** — not yet implemented in this codebase. If adopted, this section must be updated with the actual MATLAB Engine API (or file-exchange) call pattern used to invoke the Simulink model from the Python orchestration layer, including exact function/model names — do not guess at this pattern until it exists in the repo.

### Project Integration

This subsystem lives entirely in `simulation/stimulation_model.py` (or an equivalent MATLAB-invocation wrapper if that path is chosen). It is called only by `controller/adaptation.py` (to obtain suppression feedback) and by `validation/experiments/*.py` (to run comparison experiments). It has no code path to any physical actuator.

### Configuration

| Parameter | Value | Source of Truth |
|---|---|---|
| Simulation timestep | TBD | `config.simulation.timestep_s` |
| Simulated latency | TBD | `config.simulation.latency_ms` |
| Which path is active (Python vs MATLAB/Simulink) | TBD | To be recorded in `config/system_config.yaml` once decided |

### Data Contract

Input: `StimParams{amplitude, pulse_frequency_hz, pulse_width_us, duty_cycle, on_off_timing, phase}` + `tremor_state`. Output: `SimResult{achieved_suppression_pct, residual_amplitude, latency_ms, stability_warning}`. Exact field types/units match `architecture.md` → Interfaces & Data Contracts.

**Rules:**
- Never present this subsystem's output as clinically validated — always retain the "not a clinically validated physiological model" docstring/comment at the top of the module.
- Never let this module call a GPIO/DAC/hardware driver; it must remain reachable only through `simulation.apply(...)`.
- Never mix simulation-only assumptions (e.g., idealized linear suppression response) into `validation/metrics.py` reporting without labeling them as simulated, not measured-on-hardware, results.
- If/when the MATLAB/Simulink path is implemented, document the exact invocation pattern here before use — do not invent a MATLAB Engine API call pattern speculatively.

---

## Serial Studio

**Check first:** Official Serial Studio documentation

**Used by:** Visualization (development/debugging only)

**Purpose:** Provides real-time visualization of the raw ESP32/MPU6050 serial telemetry stream for hardware debugging and sampling-stability verification. Not part of the production processing pipeline.

### Project data-stream format consumed

Serial Studio reads the same CSV stream defined in the Arduino/ESP32 section: `timestamp_us,ax,ay,az,gx,gy,gz` at 115200 baud. Project-specific Serial Studio configuration (dashboard JSON project file, if used) should be stored under `firmware/serial_studio/` if/when created.

### Project Integration

Serial Studio attaches passively to the same USB serial port used by the PC-side data logger. It is a read-only consumer of the stream and has no write path back to the ESP32 in this architecture. It must never be treated as a required runtime dependency for the actual detection/controller pipeline — it is a debugging aid only, used at the `/monitor` stage described in `architecture.md`.

### Configuration

| Parameter | Value | Source of Truth |
|---|---|---|
| Serial port/baud | Matches firmware: 115200 baud | `firmware/src/serial_protocol.h` |
| Dashboard/project file | TBD (not yet created) | `firmware/serial_studio/` (if added) |

### Data Contract

Same as the Arduino/ESP32 CSV contract: `timestamp_us,ax,ay,az,gx,gy,gz`, no additional fields.

**Rules:**
- Never allow production processing code (`signal_processing/`, `ml/`, `controller/`) to depend on Serial Studio being open or running — it is purely an optional, external, human-facing debugging tool.
- Never add Serial Studio-specific formatting or extra fields to the firmware's serial output; the CSV stream must remain the single stable machine-readable format documented in Data Contracts.
- Do not treat Serial Studio as a data-logging mechanism for `data/raw/` — raw recordings for processing must come from the PC-side logger reading the same stream, not from Serial Studio exports, unless explicitly decided and documented here.

---

## Cross-Library Integration Rules

- ESP32 acquisition (Arduino-ESP32 Core + Wire) produces the defined raw IMU CSV format (`timestamp_us,ax,ay,az,gx,gy,gz`); every downstream consumer must read exactly this schema.
- Python processing (NumPy/SciPy) consumes recorded sensor data using the same documented schema whether the source is a live serial stream or a `data/raw/*.csv` file — the two paths must produce identical in-memory representations.
- Signal-processing code (NumPy/SciPy) must not depend on ML (scikit-learn) implementation details — it produces a feature vector and stops.
- ML code (scikit-learn) must consume explicitly defined features and preprocessing outputs (the pandas-assembled feature table, converted to an ordered array) — it must never reach back into `signal_processing/` internals to recompute a feature differently.
- The controller must consume ML/analysis outputs (from `ml/inference.py` and `estimation/`) through the defined `InferenceResult`/estimation dict interfaces only.
- The controller must not directly access NumPy/pandas internals or raw sensor streams — it operates on plain dataclass/dict values only.
- The stimulation simulation (SciPy `solve_ivp` or MATLAB/Simulink) must consume controller outputs through the defined `StimParams` interface only.
- Validation code (pandas, SciPy, scikit-learn metrics) must consume outputs from the pipeline (feature tables, `SimResult`, `InferenceResult`) rather than modifying internal state of `signal_processing/`, `ml/`, `controller/`, or `simulation/`.
- Visualization tools (Serial Studio, matplotlib) must consume defined output streams (the CSV format, or files in `data/`) and must not become dependencies of the processing or control pipeline — the pipeline must run correctly with no visualization tool attached.
- Do not introduce duplicate implementations of the same signal-processing operation across libraries (e.g., do not implement a second band-pass filter using pandas rolling-window math when `scipy.signal` already owns filtering).

---

## Version and Compatibility Requirements

| Dependency | Version | Runtime | Used By | Compatibility Notes |
|---|---|---|---|---|
| Arduino-ESP32 Core | TBD | ESP32 (firmware) | `firmware/` | Version not yet pinned in project files |
| Wire | Bundled with Arduino-ESP32 Core | ESP32 (firmware) | `firmware/` | Version tied to core version above |
| Python | 3.10+ | PC | All Python modules | Minimum version per `code-standards.md` |
| NumPy | TBD | PC | `signal_processing/`, `estimation/`, `simulation/` | Version not yet pinned |
| SciPy | TBD | PC | `signal_processing/`, `estimation/`, `simulation/` | Version not yet pinned |
| pandas | TBD | PC | `signal_processing/feature_extraction.py`, `ml/`, `validation/` | Version not yet pinned |
| scikit-learn | TBD | PC | `ml/` | Version not yet pinned |
| MATLAB/Simulink | TBD | PC (optional path) | `simulation/` (alternative to Python ODE) | Which path is active is itself TBD (see architecture Open Questions) |
| Serial Studio | TBD | PC (external tool) | Visualization/debugging | Optional, not a runtime dependency of the pipeline |

Pin exact versions in a `requirements.txt`/`pyproject.toml` (Python) and `platformio.ini` (firmware) as the project matures; until then, all entries above remain TBD rather than guessed.

---

## Approved Usage Patterns

- Arduino-ESP32 Core → firmware runtime, timer-driven acquisition, Serial I/O.
- Wire → MPU6050 I²C communication only, encapsulated in `mpu6050_driver.cpp/.h`.
- NumPy → numerical arrays and vectorized signal operations (calibration, filtering inputs/outputs, magnitude, RMS).
- SciPy → filtering (`butter`/`filtfilt`), spectral analysis (`welch`), and ODE simulation (`solve_ivp`).
- pandas → dataset/feature-table manipulation and file I/O, not real-time/per-sample signal processing.
- scikit-learn → classical ML models, preprocessing (`StandardScaler`), evaluation, and subject-level splitting (`GroupKFold`).
- MATLAB/Simulink or Python ODE → simulated stimulation-response modeling only, never physical actuation.
- Serial Studio → live raw-telemetry visualization/monitoring only, never a pipeline dependency.

The agent must prefer these established patterns over introducing alternative libraries for the same task.

---

## Dependency Addition Rules

- Never add a dependency without first checking whether an existing dependency (NumPy, SciPy, pandas, scikit-learn) already provides the required functionality.
- Never introduce a dependency solely for a trivial utility that can be implemented safely with the existing stack.
- Every newly approved dependency must be added to this file, with a new section following the exact structure used above.
- Document its purpose, version, integration point, and usage pattern before using it in any module.
- Do not introduce cloud services or external APIs unless explicitly required by the architecture (`architecture.md` states Database/Auth/Payments/Cloud Backend are all `None`).
- Do not introduce a new ML framework (e.g., TensorFlow, PyTorch) when the existing Python/scikit-learn pipeline can perform the required operation — `architecture.md` explicitly favors lightweight classical ML over deep learning unless proven insufficient.
- Do not replace an existing signal-processing implementation (e.g., swapping `scipy.signal.welch` for a different PSD method) without documenting the reason in this file and in `architecture.md` → Architecture Decisions.

---

## Dash

**Role:** Demonstration-only live dashboard library. Used exclusively by `scripts/run_live_demo.py` to display the real-time tremor pipeline in a browser. Not a production pipeline dependency — no module in `signal_processing/`, `ml/`, `controller/`, `simulation/`, or `validation/` imports Dash.

**Approved version:** `dash` (latest stable; brings `plotly` and `flask` as transitive dependencies).

**Why added:** Provides a browser-based live dashboard with real-time updating panels via `dcc.Interval` polling. Selected over `matplotlib` animation because it does not require a GUI toolkit (Tcl/Tk, Qt) on the host machine, runs in any browser, and produces a visually cleaner demonstration display.

**Integration point:** `scripts/run_live_demo.py` only. Imported at module level inside that script. No other script or module in the project imports Dash or Plotly.

**Architecture boundary:** The Dash app is a read-only observer of pipeline state. No Dash callback invokes any controller, simulation, or adaptation function that mutates production state. All pipeline mutations happen in the background processor thread; callbacks only read from `SharedState` under a lock.

**Threading model:** Dash runs Flask in the main thread via `app.run(debug=False, use_reloader=False)`. Pipeline threads (reader, processor) are daemon threads started before `app.run()`. Shared state is protected by `threading.Lock`; Dash callbacks acquire the lock only for the duration of a shallow copy of the fields they need.

**Usage patterns approved for this project:**

```python
# Layout
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

app = Dash(__name__, title="Tremor Demo")
app.layout = html.Div([
    dcc.Interval(id="interval-fast", interval=200, n_intervals=0),
    dcc.Graph(id="graph-raw"),
])

# Callback
@app.callback(Output("graph-raw", "figure"), Input("interval-fast", "n_intervals"))
def update(n: int) -> go.Figure:
    ...

# Launch — always use these exact flags to prevent double-process spawning
app.run(debug=False, use_reloader=False, host="127.0.0.1", port=8050)
```

**Do not use:**
- `app.run(debug=True)` — spawns a reloader subprocess that creates a second reader/processor thread.
- `dash_bootstrap_components` or any other Dash extension — not approved; style with inline `style={}` dicts instead.
- `dcc.Store` for shared state — use Python `threading.Lock` + a module-level `SharedState` instance (the standard single-worker-process pattern).
- Plotly Express (`import plotly.express as px`) — use `plotly.graph_objects` directly for full control over trace styling.

