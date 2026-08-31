# Architecture

## Stack

| Layer | Tool/Technology | Purpose |
|---|---|---|
| MCU | ESP32 | Runs firmware, acquires IMU data, streams serial telemetry |
| IMU Sensor | MPU6050 (6-axis: 3-axis accel + 3-axis gyro) | Raw wrist-motion sensing over I²C |
| Firmware Framework | Arduino / ESP-IDF | ESP32 application code, I²C driver, serial output |
| Bus Protocol | I²C (400 kHz) | ESP32 ↔ MPU6050 communication |
| Host Interface | USB Serial (115200 baud) | ESP32 → PC raw data stream |
| Telemetry Visualization | Serial Studio | Live raw-signal monitoring, hardware debugging only |
| Signal Processing | Python, NumPy, SciPy, pandas | Calibration, filtering, windowing, PSD, feature extraction |
| Filtering | SciPy Butterworth band-pass | Tremor-band (4–12 Hz) and voluntary-band (0.5–4 Hz) isolation |
| Spectral Analysis | SciPy Welch PSD | Dominant frequency, power ratio, spectral features |
| ML Framework | scikit-learn | Tremor/no-tremor classification, severity estimation |
| Candidate ML Models | Logistic Regression, SVM, Random Forest | Model comparison, smallest-model selection for ESP32 |
| Simulation | Python ODE/state-space (SciPy `solve_ivp`) or MATLAB/Simulink | Simulated tremor + stimulation-response model |
| Controller | Custom rule-based Python module | Adaptive closed-loop decision/adaptation logic |
| Data Storage | CSV / local filesystem (Parquet optional for processed windows) | Raw recordings, processed datasets, model artifacts |
| Config Management | Single YAML/JSON config file + Python dataclass loader | Central source of truth for all tunable parameters |
| Testing | pytest | Unit/integration tests for Python components |
| Database | None | Not used in current scope |
| Auth | None | Not used in current scope |
| Payments | None | Not used in current scope |
| Cloud Backend | None | Not used in current scope |

## Folder Structure

```text
tremor-system/
├── firmware/                          → ESP32 embedded code (owns hardware acquisition only)
│   ├── platformio.ini                 → Build config, board target, library deps
│   ├── src/
│   │   ├── main.cpp                   → Setup/loop, orchestrates acquisition + serial output
│   │   ├── mpu6050_driver.cpp/.h      → I²C init, register config, raw sample read
│   │   ├── sampling_timer.cpp/.h      → 100 Hz acquisition timing (hardware timer/ISR)
│   │   └── serial_protocol.cpp/.h     → CSV line formatting: timestamp_us,ax,ay,az,gx,gy,gz
│   └── test/                          → Firmware-level smoke tests (sample rate, I²C ack)
│
├── config/
│   └── system_config.yaml             → SINGLE source of truth for all tunable parameters
│                                         (sampling rate, filter band, window size, ML thresholds,
│                                          controller bounds, simulation constants)
│
├── data/
│   ├── raw/                           → Raw ESP32 CSV recordings (immutable once written)
│   │   └── <subject_id>/<session_id>/raw_stream.csv
│   ├── processed/                     → Calibrated + filtered + windowed data (derived, regenerable)
│   │   └── <subject_id>/<session_id>/windows.parquet
│   ├── features/                      → Extracted feature vectors per window
│   │   └── <subject_id>/<session_id>/features.csv
│   └── models/                        → Trained model artifacts + scalers
│       ├── scaler.pkl
│       └── model_<name>_<version>.pkl
│
├── signal_processing/                 → Owns: calibration → filtering → windowing → PSD → features
│   ├── calibration.py                 → Stationary-offset estimation, bias correction
│   ├── axis_handling.py               → Per-axis, 3-axis magnitude, strongest-axis selection
│   ├── filtering.py                   → Butterworth band-pass (tremor band, voluntary band)
│   ├── windowing.py                   → Overlapping window segmentation (1–2 s, 50% overlap)
│   ├── spectral_analysis.py           → Welch PSD, dominant frequency, tremor-band power
│   ├── time_domain.py                 → RMS, amplitude, variance
│   └── feature_extraction.py          → Assembles final feature vector per window
│
├── ml/                                 → Owns: feature vector → classification/severity/confidence
│   ├── dataset_builder.py             → Subject-level train/val/test split construction
│   ├── train.py                       → Trains + compares LogReg/SVM/RandomForest
│   ├── evaluate.py                    → Precision/recall/F1/sensitivity/FPR reporting
│   ├── model_selection.py             → Selects smallest model meeting requirements
│   └── inference.py                   → Loads model+scaler, returns (label, severity, confidence)
│
├── estimation/                        → Owns: frequency/amplitude/(later) phase estimation
│   ├── frequency_estimation.py        → Dominant peak extraction from tremor-band PSD
│   ├── amplitude_estimation.py        → RMS-based amplitude from filtered tremor signal
│   └── phase_estimation.py            → (advanced-stage placeholder, disabled by default)
│
├── controller/                        → Owns: mitigation decision + parameter adaptation ONLY
│   ├── controller_state.py            → Controller state machine (params, hysteresis, timers)
│   ├── decision_logic.py              → Confidence gating + severity-based mitigate/no-mitigate
│   ├── parameter_selection.py         → Initial stimulation parameter selection (bounded)
│   └── adaptation.py                  → Increase/decrease/maintain logic vs measured suppression
│
├── simulation/                        → Owns: simulated tremor + simulated stimulation response
│   ├── tremor_model.py                → Synthetic/replayed tremor signal generator
│   ├── stimulation_model.py           → Simulated electrical-stimulation response (ODE/state-space)
│   └── closed_loop_runner.py          → Wires controller ↔ simulation for full-loop execution
│
├── validation/                        → Owns: experiment orchestration + metric computation
│   ├── experiments/
│   │   ├── no_mitigation.py
│   │   ├── fixed_mitigation.py
│   │   └── adaptive_mitigation.py
│   ├── metrics.py                     → Suppression, residual amplitude, time-to-target, exposure
│   └── robustness_tests.py            → Noise, confidence-failure, non-stationary tremor tests
│
├── visualization/                     → Read-only observation layer (Python plots/dashboard)
│   ├── signal_plots.py
│   ├── controller_dashboard.py
│   └── validation_report.py
│
├── tests/                             → pytest suite mirroring module structure
│   ├── test_signal_processing.py
│   ├── test_ml.py
│   ├── test_controller.py
│   └── test_simulation.py
│
└── scripts/                            → CLI entrypoints tying modules together
    ├── run_acquisition_check.py       → Milestone 1: verify raw 100 Hz stream
    ├── run_pipeline_baseline.py       → Milestone 2: filtering/PSD visual verification
    ├── run_training.py                → Trains and evaluates ML models
    └── run_closed_loop_simulation.py  → Runs full Detect→...→Adapt loop in simulation
```

## System Boundaries

| Component | Owns | Must NOT Do |
|---|---|---|
| `firmware/` | I²C acquisition, 100 Hz sampling, timestamping, raw serial CSV output | No filtering, no feature extraction, no ML inference, no stimulation logic |
| `signal_processing/` | Calibration, filtering, windowing, PSD, feature extraction | No ML training/inference, no controller decisions, no direct hardware access |
| `ml/` | Training, evaluation, subject-level splitting, inference (label/severity/confidence) | No raw signal filtering, no controller/parameter logic, no simulation |
| `estimation/` | Frequency/amplitude/(later) phase calculation from processed signal | No classification, no mitigation decisions |
| `controller/` | Mitigation decision, stimulation parameter selection, adaptation | No raw sensor access, no signal processing, no ML logic, no direct stimulation output |
| `simulation/` | Simulated tremor response, simulated stimulation response | No real hardware actuation, no ML/controller logic |
| `validation/` | Experiment orchestration, metric computation across strategies | No modification of production `signal_processing/`, `ml/`, or `controller/` logic |
| `visualization/` | Read-only display of data produced by other modules | No writes to `data/`, no control commands to controller or simulation |
| `config/` | Single source of truth for tunable parameters | No component may hardcode a duplicate constant |

**Boundary rule:** data flows strictly one direction through the pipeline (firmware → signal_processing → ml/estimation → controller → simulation → validation/visualization). No downstream module writes back into an upstream module's internal state; feedback (e.g., measured suppression) is passed explicitly as a function argument/return value, never via shared globals.

## Data Flow

```text
# Full closed-loop data flow (single window cycle)

MPU6050 raw registers
   → firmware/mpu6050_driver.cpp (I²C read, 14-byte block)
   → firmware/serial_protocol.cpp (CSV: timestamp_us,ax,ay,az,gx,gy,gz)
   → USB Serial @ 115200 baud
   → data/raw/<subject>/<session>/raw_stream.csv  (or live stdin stream)
   → signal_processing/calibration.py (offset correction)
   → signal_processing/axis_handling.py (per-axis / magnitude / strongest-axis)
   → signal_processing/filtering.py (gravity/baseline removal + Butterworth band-pass)
   → signal_processing/windowing.py (1–2s window, 50% overlap)
   → signal_processing/spectral_analysis.py (Welch PSD)
   → signal_processing/time_domain.py (RMS/variance)
   → signal_processing/feature_extraction.py (assembled feature vector)
   → ml/inference.py (label, severity, confidence)
   → estimation/frequency_estimation.py (dominant frequency from tremor-band PSD)
   → estimation/amplitude_estimation.py (RMS-based amplitude)
   → [estimation/phase_estimation.py — advanced stage, optional/disabled]
   → controller/decision_logic.py (confidence gate → mitigate? yes/no)
   → controller/parameter_selection.py (bounded stim params: amplitude, freq, pulse width, duty cycle)
   → simulation/stimulation_model.py (applies simulated stimulation to simulated tremor)
   → simulation/tremor_model.py (produces post-mitigation tremor signal)
   → validation/metrics.py or controller/adaptation.py (measure suppression = before vs after power/RMS)
   → controller/adaptation.py (compare to target_suppression → increase/decrease/maintain params)
   → [loop back to controller/parameter_selection.py for next cycle]
   → visualization/controller_dashboard.py (read-only display of all of the above)
```

## Storage

| Storage Type | Path/Location | Contents | Producer | Consumer | Access Rules |
|---|---|---|---|---|---|
| Raw IMU recordings | `data/raw/<subject_id>/<session_id>/raw_stream.csv` | timestamp_us, ax, ay, az, gx, gy, gz | `firmware/` via serial logger | `signal_processing/calibration.py` | Immutable/append-only; never overwritten |
| Processed windows | `data/processed/<subject_id>/<session_id>/windows.parquet` | Calibrated, filtered, windowed signal segments | `signal_processing/windowing.py` | `signal_processing/spectral_analysis.py`, `ml/dataset_builder.py` | Regenerable from raw; safe to delete/rebuild |
| Feature vectors | `data/features/<subject_id>/<session_id>/features.csv` | Tremor-band power, total power, ratio, dominant freq, RMS, variance, etc. | `signal_processing/feature_extraction.py` | `ml/train.py`, `ml/inference.py` | Regenerable; must retain subject_id/session_id for split integrity |
| Trained models + scalers | `data/models/model_<name>_<version>.pkl`, `data/models/scaler.pkl` | Serialized sklearn models/scalers | `ml/train.py` | `ml/inference.py` | Versioned filenames; never silently overwritten |
| Simulation results | `data/simulation/<experiment_id>/` | Simulated tremor/stimulation time series, suppression traces | `simulation/closed_loop_runner.py` | `validation/metrics.py`, `visualization/` | Tagged by experiment type (no-mitigation/fixed/adaptive) |
| Validation outputs | `data/validation/<experiment_id>/report.json` | Precision/recall/F1, suppression %, latency, duty cycle, exposure, stability | `validation/metrics.py` | `visualization/validation_report.py` | Read-only after generation; regenerate rather than edit |
| Configuration | `config/system_config.yaml` | All tunable parameters (single source of truth) | Manually edited / experiment scripts | All modules via config loader | Never hardcode duplicates elsewhere |

## Key Integration Patterns

**ESP32 ↔ MPU6050**
- I²C, SDA=GPIO21, SCL=GPIO22, clock=400 kHz.
- Firmware reads the full 14-byte measurement block per sample (accel×3, temp, gyro×3).
- Sampling triggered by a hardware timer targeting 100 Hz; DLPF=`0x03`, accel range ±2g, gyro range ±250°/s.

**ESP32 ↔ PC**
- USB Serial, 115200 baud, one CSV line per sample: `timestamp_us,ax,ay,az,gx,gy,gz`.
- No acknowledgement/handshake protocol in current scope; PC-side logger treats malformed lines as dropped samples (see Error Handling).
- Serial Studio attaches to the same serial port as a passive reader for live telemetry only; it must not write commands back to the ESP32 in current scope.

**Python signal-processing pipeline ↔ ML model**
- Interface is a fixed-length numeric feature vector (see Interfaces & Data Contracts) plus `subject_id`/`session_id`/`window_id` metadata for traceability.
- Signal processing has no knowledge of which ML model consumes the vector; ML has no knowledge of how the vector was derived beyond the documented feature schema.

**ML ↔ controller**
- Controller consumes only: `{label: bool, severity: float, confidence: float, dominant_frequency_hz: float, amplitude: float}`.
- Controller never receives raw signal, PSD, or feature vector directly — this is a hard boundary preventing the controller from re-implementing signal/ML logic.

**Controller ↔ stimulation simulation**
- Controller calls `simulation.apply(params: StimParams) -> SimResult` where `StimParams = {amplitude, pulse_frequency_hz, pulse_width_us, duty_cycle, on_off_timing, phase}` (phase optional/None in initial stage).
- `SimResult` returns `{post_mitigation_signal, achieved_suppression_pct, latency_ms}`.
- This call is synchronous within a simulation timestep; no direct hardware I/O occurs here.

**Visualization/dashboard ↔ data source**
- Visualization reads from `data/` artifacts and/or in-memory objects passed explicitly by the calling script; it never queries firmware or controller state directly, and never issues control calls.

## Invariants

1. Sampling rate is fixed at a **target of 100 Hz**; all window-size and filter-order calculations must reference `config/system_config.yaml`, never a hardcoded literal.
2. Raw signal units: accelerometer in **g** (post ±2g-range conversion), gyroscope in **°/s** (post ±250°/s-range conversion). All downstream modules must document/convert units explicitly — never assume raw LSB values.
3. Tremor band is **4–12 Hz**; voluntary-movement band is **0.5–4 Hz**. Both are configurable, but the default values live only in `config/system_config.yaml`.
4. Windowing uses **1–2 s overlapping windows at 50% overlap** by default; any change must go through config, not inline constants.
5. Welch PSD is the default spectral method; raw FFT alone must not replace it without an explicit config flag.
6. ML train/validation/test splits **must be subject-level**, never window-level, to prevent leakage across overlapping windows from the same subject.
7. `ml/inference.py` must not receive or use raw sensor data — only the documented feature vector schema.
8. The controller (`controller/`) must **never** access raw sensor data, PSD, or feature vectors directly, and must **never** implement signal-processing or ML logic itself.
9. Stimulation is **simulation-only**. No component in this architecture may issue a command to a physical stimulation actuator. Any future physical-actuator integration requires a new, explicitly reviewed module boundary.
10. All controller-issued stimulation parameters must remain within the bounds defined in `config/system_config.yaml` (amplitude, pulse frequency, pulse width, duty cycle) — no unbounded adaptation.
11. Controller adaptation rate is capped per cycle (`max_delta_per_step` in config) and gated by hysteresis to prevent rapid ON/OFF oscillation.
12. Controller mitigation decisions must be gated by detection confidence; low-confidence detections must not trigger stimulation-parameter changes (see Error Handling).
13. Each module owns exactly one stage of the pipeline as defined in System Boundaries; no module may duplicate another module's responsibility.
14. All tunable/experimental parameters must be read from `config/system_config.yaml`; no module may hardcode a duplicate constant.
15. Phase estimation is disabled/optional by default; the controller and simulation must function correctly with `phase=None`.
16. Validation experiments (`validation/`) must not modify the production signal-processing, ML, or controller code paths — they call the same production functions unmodified.

## Hardware Architecture

- **ESP32**: sole compute unit on the wearable; owns sampling timing, I²C transactions, and serial output. No stimulation hardware is connected or driven by the ESP32 in the current scope.
- **MPU6050**: 6-axis IMU (3-axis accelerometer + 3-axis gyroscope), connected via I²C at SDA=GPIO21, SCL=GPIO22, 400 kHz clock. Configured for ±2g accel range, ±250°/s gyro range, DLPF=`0x03`, nominal 100 Hz output rate.
- **Power/Communication**: ESP32 powered and data-connected via USB during current development phase (no battery/wireless telemetry defined yet — see Open Questions).
- **Boundary**: sensing hardware (ESP32 + MPU6050) is physically and logically isolated from the stimulation subsystem, which exists only as software simulation (`simulation/stimulation_model.py`). No GPIO, DAC, or actuator driver code exists in `firmware/` for stimulation output.

## Firmware Architecture

- **Initialization**: configure I²C bus, verify MPU6050 WHO_AM_I register, write DLPF/range/sample-rate-divider registers.
- **Acquisition loop**: hardware-timer-driven read at 100 Hz target; each read pulls the full 14-byte block (ax, ay, az, temp, gx, gy, gz) in one I²C transaction.
- **Timestamping**: each sample tagged with a monotonic `timestamp_us` (e.g., `micros()`), not a fixed-interval assumption, so downstream code can detect timing drift/jitter.
- **Buffering**: minimal — samples are serialized and transmitted immediately; no on-device windowing/filtering buffer in the current implementation.
- **Serial output format**: one CSV line per sample, `timestamp_us,ax,ay,az,gx,gy,gz`, raw integer/float register values (unit conversion happens on the PC side in `signal_processing/calibration.py`, not on-device).
- **Processing that must remain on the PC**: all calibration, filtering, windowing, PSD, feature extraction, ML inference, frequency/phase estimation, controller logic, and stimulation simulation. The ESP32 performs **no** signal processing or inference in the initial implementation; on-device inference is a later milestone contingent on model-size validation (see Deployment Architecture).

## Signal Processing Architecture

```text
calibration.py          → stationary-offset estimation, bias correction
      ↓
axis_handling.py         → choose representation: per-axis | 3-axis magnitude | strongest-axis
      ↓
filtering.py              → gravity/baseline removal, Butterworth band-pass (tremor + voluntary bands)
      ↓
windowing.py               → overlapping windows (size/overlap from config)
      ↓
spectral_analysis.py        → Welch PSD → tremor-band power, total power, power ratio, dominant frequency
      ↓
time_domain.py                → RMS, amplitude, variance
      ↓
feature_extraction.py          → assembles final feature vector (schema in Interfaces section)
```

- Filter type (Butterworth), band edges (4–12 Hz / 0.5–4 Hz), window length, and overlap are all **experimentally tunable** and must be read from `config/system_config.yaml`, never hardcoded in the module files.
- `axis_handling.py` must support running all three representations (per-axis, magnitude, strongest-axis) side-by-side during development for comparison; the "chosen" strategy is itself a config value, not a permanent code branch removal.

## ML Architecture

```text
feature_extraction.py (features.csv)
      ↓
dataset_builder.py        → subject-level train/val/test split (GroupKFold or manual subject partition)
      ↓
train.py                    → fits LogisticRegression, SVM, RandomForest candidates
      ↓
evaluate.py                   → precision, recall, F1, sensitivity, false-positive rate per model
      ↓
model_selection.py              → selects smallest model meeting the project's detection requirements
      ↓
inference.py                      → loads model + scaler.pkl → returns (label, severity, confidence)
```

- Subject-level splitting is enforced in `dataset_builder.py` via a grouping key (`subject_id`); window-level shuffling is explicitly disallowed (see Invariant 6).
- Severity is derived either as a model output (e.g., regression head) or a rule-based mapping from tremor-band power/amplitude — the exact method is an Open Question but must be isolated behind `inference.py`'s return contract so the controller is unaffected by the choice.
- Confidence is taken from model probability output (`predict_proba`) for classifiers that support it; models without native probability output must be excluded or wrapped (e.g., `CalibratedClassifierCV`).

## Frequency & Phase Estimation

- **Frequency**: `estimation/frequency_estimation.py` extracts the dominant peak from the tremor-band Welch PSD (owned by `signal_processing/spectral_analysis.py`, consumed here). This is available from the first implementation milestone onward.
- **Amplitude**: `estimation/amplitude_estimation.py` computes RMS (or equivalent) from the filtered tremor-band time-domain signal.
- **Phase**: `estimation/phase_estimation.py` is an advanced-stage module, disabled by default. It is not required for the initial controller, which operates on frequency + severity + confidence only. When enabled, it must expose the same optional-output contract (`phase: Optional[float]`) so the controller and simulation degrade gracefully to `phase=None`.
- Ownership: estimation logic lives entirely in `estimation/`, not in `ml/` or `controller/`.

## Adaptive Controller

```text
Input: {confidence, severity, dominant_frequency_hz, amplitude, [phase]}
      ↓
decision_logic.py    → confidence >= threshold? severity >= mitigation_threshold? → mitigate: bool
      ↓ (if mitigate)
parameter_selection.py → initial bounded StimParams {amplitude, pulse_frequency_hz, pulse_width_us,
                          duty_cycle, on_off_timing, phase?}
      ↓
[handoff to simulation/stimulation_model.py — see Stimulation Simulation]
      ↓ (returns achieved_suppression_pct)
adaptation.py          → compare achieved vs target_suppression:
                             if under target: increase params (bounded, capped rate)
                             if over target: decrease params (minimize exposure objective)
                             if within suppression tolerance band: maintain
      ↓
Output: updated StimParams for next cycle + controller_state snapshot (for logging/dashboard)
```

- **State owned**: current stimulation parameters, adaptation history, hysteresis band state, last decision timestamp.
- **Confidence gating**: below-threshold confidence forces `mitigate=False` regardless of severity (see Error Handling).
- **Bounds**: all parameters clamped to `[min, max]` ranges defined per-parameter in `config/system_config.yaml`.
- **Hysteresis**: mitigation on/off decision requires crossing distinct enter/exit thresholds to prevent chatter.
- **Adaptation rate cap**: `max_delta_per_step` limits how much any parameter can change in one cycle.
- **Objective**: minimize stimulation exposure/duty cycle subject to achieving `target_suppression`.

## Stimulation Simulation

- **Interface**: `simulation.apply(params: StimParams, tremor_state: TremorState) -> SimResult`.
- **StimParams**: `{amplitude, pulse_frequency_hz, pulse_width_us, duty_cycle, on_off_timing, phase: Optional[float]}` — identical schema to what the controller emits.
- **Tremor-response model**: implemented as an ODE/state-space system (Python `scipy.integrate.solve_ivp` or MATLAB/Simulink) representing the relationship between stimulation parameters and tremor amplitude reduction. Explicitly **not** a clinically validated physiological model — used only for controller development/validation.
- **SimResult**: `{post_mitigation_signal, achieved_suppression_pct, residual_amplitude, latency_ms}`.
- **Latency**: the simulation must model a configurable detection-to-actuation latency (`config/system_config.yaml: simulation.latency_ms`) so the controller is validated against realistic timing, not zero-latency idealization.
- **Boundary**: this module has no code path to any physical GPIO/DAC/actuator driver. Any future physical-stimulation integration must be a new, separately reviewed module — this boundary is a hard architectural constraint (see Invariant 9, Security & Safety Boundaries).

## Validation Architecture

- Validation experiments (`validation/experiments/{no_mitigation, fixed_mitigation, adaptive_mitigation}.py`) each call the **same** production `controller/` and `simulation/` code with different configurations — they do not fork or duplicate controller logic.
- `no_mitigation.py`: runs the simulated tremor model with the controller decision forced to `mitigate=False`.
- `fixed_mitigation.py`: runs with a static, non-adapting `StimParams` set (adaptation.py bypassed).
- `adaptive_mitigation.py`: runs the full closed loop unmodified.
- Ground truth for frequency/amplitude validation comes from the synthetic tremor generator (`simulation/tremor_model.py`), which can inject known frequency/amplitude for error measurement.
- Subject-level train/val/test separation from the ML pipeline is reused (not re-derived) when validation involves recorded (non-synthetic) data.
- Outputs land in `data/validation/<experiment_id>/report.json` and are never used to retroactively edit the underlying pipeline.

## Dashboard / Visualization

- Current interface is **not a web app**; it is Serial Studio (raw telemetry) plus Python plotting/reporting scripts (`visualization/`).
- Serial Studio: read-only consumer of the live USB serial stream; used exclusively for `/monitor`-equivalent hardware debugging (raw ax/ay/az/gx/gy/gz, sampling stability). It has no write path back to the ESP32 in this architecture.
- Python visualization (`signal_plots.py`, `controller_dashboard.py`, `validation_report.py`): read-only consumers of `data/processed/`, `data/features/`, `data/simulation/`, `data/validation/`, and in-memory objects passed by orchestration scripts. No visualization module may call `controller/` or `simulation/` functions that mutate state — display only.

## Configuration

| Parameter | Source of Truth | Default/Experimental Value | Unit | Consumed By |
|---|---|---|---|---|
| Sampling rate | `system_config.yaml: sensor.sample_rate_hz` | 100 | Hz | firmware, signal_processing |
| I²C clock | `system_config.yaml: sensor.i2c_clock_hz` | 400000 | Hz | firmware |
| Accel range | `system_config.yaml: sensor.accel_range_g` | ±2 | g | firmware, calibration |
| Gyro range | `system_config.yaml: sensor.gyro_range_dps` | ±250 | °/s | firmware, calibration |
| Tremor band | `system_config.yaml: signal.tremor_band_hz` | [4, 12] | Hz | filtering, spectral_analysis |
| Voluntary band | `system_config.yaml: signal.voluntary_band_hz` | [0.5, 4] | Hz | filtering |
| Filter order | `system_config.yaml: signal.filter_order` | TBD (experimental) | — | filtering |
| Window length | `system_config.yaml: signal.window_length_s` | 1–2 (experimental) | s | windowing |
| Window overlap | `system_config.yaml: signal.window_overlap_pct` | 50 | % | windowing |
| Axis strategy | `system_config.yaml: signal.axis_strategy` | TBD (per-axis/magnitude/strongest) | — | axis_handling |
| ML detection threshold | `system_config.yaml: ml.confidence_threshold` | TBD (experimental) | probability | controller decision_logic |
| Mitigation severity threshold | `system_config.yaml: controller.severity_threshold` | TBD | severity units | controller decision_logic |
| Target suppression | `system_config.yaml: controller.target_suppression_pct` | 50 (example, tunable) | % | controller adaptation |
| Suppression tolerance band | `system_config.yaml: controller.suppression_tolerance_pct` | TBD (experimental) | % | controller adaptation |
| Hysteresis band | `system_config.yaml: controller.hysteresis_pct` | TBD | % | controller decision_logic |
| Max adaptation rate | `system_config.yaml: controller.max_delta_per_step` | TBD | per-parameter units | controller adaptation |
| Stim parameter bounds | `system_config.yaml: controller.param_bounds.*` | TBD per parameter | varies | controller parameter_selection |
| Simulation latency | `system_config.yaml: simulation.latency_ms` | TBD (experimental) | ms | simulation |
| Simulation timestep | `system_config.yaml: simulation.timestep_s` | TBD | s | simulation |

## Interfaces & Data Contracts

**ESP32 serial CSV (raw sample):**
```text
timestamp_us:int, ax:float, ay:float, az:float, gx:float, gy:float, gz:float
```

**Processed window record:**
```python
{
  "subject_id": str,
  "session_id": str,
  "window_id": int,
  "start_timestamp_us": int,
  "end_timestamp_us": int,
  "signal": np.ndarray,   # shape: (window_samples, n_channels)
}
```

**Feature vector (ML input):**
```python
{
  "subject_id": str, "session_id": str, "window_id": int,
  "tremor_band_power": float,
  "total_power": float,
  "power_ratio": float,
  "dominant_frequency_hz": float,
  "rms_amplitude": float,
  "variance": float,
  "spectral_entropy": float,
  "accel_magnitude": float,
  "gyro_magnitude": float,
}
```

**ML inference output (controller input):**
```python
{
  "label": bool,               # tremor / no-tremor
  "severity": float,           # 0.0–1.0 or defined scale
  "confidence": float,         # 0.0–1.0
  "dominant_frequency_hz": float,
  "amplitude": float,
  "phase": float | None,       # optional, advanced stage
}
```

**Controller output (simulation input):**
```python
StimParams = {
  "amplitude": float,
  "pulse_frequency_hz": float,
  "pulse_width_us": float,
  "duty_cycle": float,         # 0.0–1.0
  "on_off_timing": tuple,      # (on_ms, off_ms)
  "phase": float | None,
}

TremorState = {
      "y0": np.ndarray,             # shape (2,), initial position and velocity
      "duration_s": float,
      "timestep_s": float,          # simulation integration max_step
      "signal": np.ndarray,         # shape (n_samples,), synthetic tremor in g
      "sample_rate_hz": float,
}
```

**Simulation output (controller/validation input):**
```python
SimResult = {
  "post_mitigation_signal": np.ndarray,
  "achieved_suppression_pct": float,
  "residual_amplitude": float,
  "latency_ms": float,
}
```

## Error Handling & Failure Modes

| Failure | Detecting Component | Behavior |
|---|---|---|
| I²C read failure / NACK | `firmware/mpu6050_driver.cpp` | Skip sample, log error over serial as a distinct marker line, retry next cycle |
| Malformed/dropped serial line | PC-side logger (pre-`signal_processing/`) | Discard line, do not attempt partial parse; log gap in `timestamp_us` |
| Missing/insufficient samples in a window | `signal_processing/windowing.py` | Discard incomplete window; do not zero-pad silently |
| Low ML confidence | `controller/decision_logic.py` | Force `mitigate=False`; log as a distinct "low-confidence, no action" state, not an error |
| Frequency estimation fails (flat/no clear peak) | `estimation/frequency_estimation.py` | Return `dominant_frequency_hz=None`; controller treats as low-confidence input |
| Simulation instability (diverging ODE state) | `simulation/stimulation_model.py` | Clamp state, flag `stability_warning=True` in `SimResult`, and force controller `maintain` on next cycle |
| Downstream component unavailable (e.g., model file missing) | `ml/inference.py` | Raise explicit exception at startup (fail fast), not a silent fallback prediction |

## Performance & Timing Constraints

| Stage | Budget/Requirement |
|---|---|
| Sampling interval | ~10 ms (100 Hz target) |
| Serial transmission per sample | Must not block acquisition timer beyond 1 sample period |
| Windowing delay | ≤ window_length_s × (1 − overlap) between successive window outputs |
| Filtering | Must process one window well within the window's real-time duration for eventual real-time feasibility |
| ML inference | Must complete within the window cadence budget on target hardware (PC now; ESP32-class budget for future deployment) |
| Controller response time | One decision cycle per completed window; no sub-window control updates in current scope |
| Simulation timestep | Configurable (`simulation.timestep_s`); must be ≤ smallest relevant tremor/stimulation time constant |
| ESP32 memory | Firmware buffer limited to single-sample transmission; no multi-window buffering in current scope |

## Security & Safety Boundaries

- Stimulation remains **simulation-only**; no component in this codebase has a code path to a physical actuator, DAC, or GPIO stimulation driver.
- Only `controller/parameter_selection.py` and `controller/adaptation.py` are permitted to produce `StimParams`; no other module may construct or emit stimulation parameters.
- All `StimParams` values are validated against configured bounds before being passed to `simulation/`; out-of-bounds values must raise an error rather than being silently clamped without logging.
- Any future integration of a physical stimulation actuator requires a new, explicitly reviewed hardware/firmware module and is out of scope for this architecture (see Features Out of Scope in the project overview).
- Visualization and validation modules are strictly read-only with respect to controller/simulation state — they cannot issue control commands.

## Testing Strategy

| Layer | What It Validates | Artifacts Produced |
|---|---|---|
| Firmware unit/smoke tests | Sample rate stability, I²C ack, CSV format correctness | Pass/fail log, sample-rate report |
| Signal-processing unit tests | Filter frequency response, windowing correctness, PSD sanity checks | `pytest` results |
| ML tests | Subject-level split integrity (no leakage), metric computation correctness | `pytest` results, evaluation report |
| Controller tests | Bounds enforcement, hysteresis behavior, adaptation-rate cap, confidence gating | `pytest` results, controller state traces |
| Simulation tests | ODE stability under nominal/edge parameters, latency modeling correctness | `pytest` results |
| Integration tests | Full window → feature → ML → controller → simulation cycle, end-to-end | Integration log, sample closed-loop trace |
| End-to-end validation | No-mitigation vs fixed vs adaptive comparison, robustness under noise | `data/validation/<experiment_id>/report.json` |

## Observability & Logging

Every pipeline run logs, at minimum:
- Raw sample timestamps and values (or a reference to the raw CSV file used)
- Filtered/windowed signal summaries (not full raw arrays, to bound log size)
- Extracted feature vectors per window
- ML prediction: label, severity, confidence
- Estimated frequency (and phase, when enabled)
- Controller decision (mitigate/no-mitigate), selected/adapted `StimParams`, hysteresis state
- Simulation `SimResult` (achieved suppression, residual amplitude, latency, stability warnings)
- All errors/failure-mode events listed in Error Handling & Failure Modes, tagged with the originating component

## Dependencies

| Dependency | Version/Environment | Used By | Purpose | Required/Optional |
|---|---|---|---|---|
| Arduino-ESP32 / ESP-IDF | Per PlatformIO board target | `firmware/` | ESP32 build/runtime | Required |
| MPU6050 I²C library or custom driver | Project-vendored | `firmware/` | Register-level IMU access | Required |
| Python | 3.10+ | All Python modules | Runtime | Required |
| NumPy | Latest stable | `signal_processing/`, `simulation/` | Array math | Required |
| SciPy | Latest stable | `signal_processing/`, `estimation/`, `simulation/` | Filtering, Welch PSD, ODE solving | Required |
| pandas | Latest stable | `signal_processing/`, `ml/`, `validation/` | Tabular data handling | Required |
| scikit-learn | Latest stable | `ml/` | Classical ML models, metrics, splitting | Required |
| MATLAB/Simulink | Institution-licensed | `simulation/` (alternative to Python ODE) | Stimulation-response modeling | Optional (alternative path) |
| Serial Studio | Latest stable | Telemetry visualization | Live raw-signal monitoring | Optional (debugging aid) |
| pytest | Latest stable | `tests/` | Automated testing | Required (dev) |
| PyYAML | Latest stable | `config/` loader | Parsing `system_config.yaml` | Required |
| matplotlib | Latest stable | `visualization/` | Plotting | Required |

## Deployment Architecture

| Component | Runs On (Current) | Runs On (Future) |
|---|---|---|
| `firmware/` | ESP32 board (USB-powered, tethered to PC) | Same ESP32, potentially with on-device inference added |
| `signal_processing/`, `ml/`, `estimation/` | PC (development environment) | Signal processing/inference may partially migrate to ESP32 pending resource validation |
| `controller/`, `simulation/` | PC (development environment) | Remains PC-side simulation-only unless a future physical-actuator project is scoped |
| `validation/`, `visualization/` | PC (development environment) | Remains PC-side |
| Data storage | Local filesystem | Local filesystem (no cloud backend planned in current scope) |

## Development vs Production Boundary

- **Experimental/tunable now**: filter order, window length, axis strategy, ML thresholds, controller thresholds/hysteresis/adaptation rate, simulation latency/timestep — all isolated in `config/system_config.yaml` and marked TBD until experimentally determined.
- **Simulation assumptions**: the entire `simulation/stimulation_model.py` is explicitly a development/validation tool, not a physiological ground truth; it must never be presented as clinically validated.
- **Temporary debugging tools**: Serial Studio and ad hoc plotting scripts in `visualization/` are development aids, not part of the eventual production/deployment pipeline.
- **Migration candidates**: `signal_processing/filtering.py`, `windowing.py`, and `ml/inference.py` (for the selected lightweight model) are the modules most likely to be ported onto the ESP32 in a future milestone; they should therefore avoid PC-only dependencies (e.g., pandas) in their core computational paths where practical.

## Architecture Decisions

| Decision | Chosen Approach | Alternatives Considered | Reason | Status |
|---|---|---|---|---|
| Sampling rate | 100 Hz target | 50 Hz, 200 Hz | Sufficient for 4–12 Hz tremor band with margin; balances ESP32/serial bandwidth | Decided |
| Tremor band | 4–12 Hz | 3–15 Hz, disease-specific bands | Standard physiological tremor range; configurable if needed | Decided (tunable) |
| Spectral method | Welch PSD | Raw FFT only | Reduces variance, more robust dominant-frequency estimate | Decided |
| ML approach | Lightweight classical ML (LogReg/SVM/RandomForest) | Deep learning | Must fit ESP32 deployment constraints; classical models sufficient for initial detection task | Decided |
| Stimulation approach | Simulation-first (ODE/state-space) | Direct physical stimulation | Safety, validation-before-hardware requirement | Decided |
| Controller design | Rule-based adaptive control | Reinforcement learning | Interpretability, bounded/safe behavior, easier validation | Decided |
| Data storage | CSV/local filesystem | Database (SQL/NoSQL) | No multi-user/production requirement in current scope | Decided |
| Split strategy | Subject-level train/val/test | Window-level random split | Prevents data leakage across overlapping windows | Decided |

## Open Questions / TBDs

1. Final Butterworth filter order for tremor-band and voluntary-band filters.
2. Final window length within the 1–2 s range and confirmation of 50% overlap as optimal.
3. Final axis-handling strategy (per-axis vs 3-axis magnitude vs strongest-axis) after comparative evaluation.
4. Final ML model selection among Logistic Regression, SVM, and Random Forest, and its ESP32 resource footprint.
5. Controller gains: exact `severity_threshold`, `confidence_threshold`, `hysteresis_pct`, and `max_delta_per_step` values.
6. Phase-estimation method and the point at which it is enabled in the controller loop.
7. Exact stimulation-parameter bounds (`param_bounds.*`) for amplitude, pulse frequency, pulse width, duty cycle.
8. Simulation latency and timestep values representative of a realistic detection-to-actuation delay.
9. How much of the signal-processing/inference pipeline is ultimately allocated to the ESP32 vs remaining PC-side, and the associated latency budget.
10. Final visualization/dashboard choice beyond Serial Studio + ad hoc Python plots (e.g., whether a persistent dashboard app is built).
