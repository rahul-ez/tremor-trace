# TremorTrace: Closed-Loop Hand Tremor Detection & Adaptive Suppression System

[![Tests](https://img.shields.io/badge/pytest-230%20passed%20%7C%201%20skipped-brightgreen)](#testing--verification)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Embedded](https://img.shields.io/badge/Hardware-ESP32%20%2B%20MPU6050-orange)](#hardware--firmware)
[![Stage](https://img.shields.io/badge/Phase-Phase%208%20(Closed--Loop%20Integration)-yellow)](#current-implementation-status)

**TremorTrace** is a research project designing a wearable, adaptive closed-loop system for hand tremor detection, characterization, and simulated electrical stimulation suppression. The system integrates embedded sensing on an **ESP32 with an MPU6050 6-axis IMU** (100 Hz sampling), a classical DSP and lightweight Machine Learning pipeline, and a bounded adaptive closed-loop controller interacting with a simulated stimulation-response dynamical model.

> [!IMPORTANT]
> **Simulation-First Research Scope:** This project is currently operating in a **simulation-first** research phase. Electrical stimulation is simulated via state-space / Ordinary Differential Equation (ODE) models to quantitatively validate detection, parameter selection, and adaptation dynamics. **No physical electrical current is applied to human subjects.**

---

## Table of Contents

- [System Architecture & Data Pipeline](#system-architecture--data-pipeline)
- [Current Implementation Status](#current-implementation-status)
  - [Completed Phases (Phases 1–7)](#completed-phases-phases-17)
  - [Test Suite & Health](#test-suite--health)
- [Remaining Work & Research Roadmap](#remaining-work--research-roadmap)
  - [Phase 8 — Closed-Loop Integration (Current)](#phase-8--closed-loop-integration-current)
  - [Phase 9 — Validation, Benchmarking & Robustness](#phase-9--validation-benchmarking--robustness)
  - [Phase 10 — Visualization & Research Dashboards](#phase-10--visualization--research-dashboards)
  - [Phase 11 — Embedded / On-Device Deployment](#phase-11--embedded--on-device-deployment)
  - [Phase N — Final System Verification](#phase-n--final-system-verification)
- [Repository Structure](#repository-structure)
- [Getting Started & Usage](#getting-started--usage)
  - [Environment Setup](#environment-setup)
  - [Hardware Telemetry & Data Logging](#hardware-telemetry--data-logging)
  - [Signal Processing Baseline](#signal-processing-baseline)
  - [Running Automated Tests](#running-automated-tests)
- [Key Architectural Contracts & Invariants](#key-architectural-contracts--invariants)
- [License & Research Disclaimer](#license--research-disclaimer)

---

## System Architecture & Data Pipeline

The closed-loop control flow follows the continuous paradigm:
$$\text{Detect} \longrightarrow \text{Analyze} \longrightarrow \text{Decide} \longrightarrow \text{Mitigate} \longrightarrow \text{Measure} \longrightarrow \text{Adapt} \longrightarrow \text{Repeat}$$

```
                ┌────────────────────────────────┐
                │     MPU6050 (6-Axis IMU)       │
                │  Accelerometer & Gyroscope     │
                └───────────────┬────────────────┘
                                │ I²C (400 kHz)
                                ▼
                ┌────────────────────────────────┐
                │          ESP32 MCU             │
                │ 100 Hz Timer-Driven Sampling   │
                │ Microsecond-Accurate Streaming │
                └───────────────┬────────────────┘
                                │ USB Serial (115200 baud)
                                ▼
                ┌────────────────────────────────┐
                │   Signal Processing & DSP      │
                │ Calibration (Offset Removal)   │
                │ Baseline / Gravity Removal     │
                │ Butterworth Bandpass (4–12 Hz) │
                │ 1–2s Windowing (50% Overlap)   │
                │ Welch PSD & Spectral Features  │
                └───────────────┬────────────────┘
                                │ Feature Vector
                                ▼
                ┌────────────────────────────────┐
                │    Lightweight ML Classifier   │
                │ LogReg / SVM / Random Forest   │
                │ (Tremor vs Voluntary Movement) │
                │ Output: Label, Severity, Conf. │
                └───────────────┬────────────────┘
                                │ Estimation: Dominant Freq & Amplitude
                                ▼
                ┌────────────────────────────────┐
                │   Adaptive Closed-Loop Engine  │
                │ Confidence Gating (> 0.6)      │
                │ Hysteresis-Gated Thresholding  │
                │ Bounded Parameter Selection    │
                └───────────────┬────────────────┘
                                │ StimParams (Amplitude, Freq, PW, Duty)
                                ▼
                ┌────────────────────────────────┐
                │ Stimulation Simulation (ODE)   │
                │ State-space tremor-response    │
                │ Realistic Latency Modeling     │
                └───────────────┬────────────────┘
                                │ Post-mitigation Signal & Suppression %
                                ▼
                ┌────────────────────────────────┐
                │     Adaptation & Feedback      │
                │ Compares vs Target Suppression │
                │ Minimizes Total Exposure       │
                └───────────────┴────────────────┘
```

---

## Current Implementation Status

The project is structured into 12 rigorous vertical phases spanning 60 feature gates. As of **August 2026**, **Phases 1 through 7 (Features 01–40)** are fully implemented and verified.

### Completed Phases (Phases 1–7)

#### ✅ Phase 1: Foundation (Features 01–04)
- Full architectural directory skeleton with strict package separation.
- Single source of truth configuration engine (`config/system_config.yaml`) with typed loader dataclasses (`tremor_system/config.py`).
- Universal data contracts (`StimParams`, `SimResult`, `InferenceResult`, `TremorState`).
- Standardized pytest testing harness.

#### ✅ Phase 2: Embedded Acquisition (Features 05–09)
- Hardware-timer-driven 100 Hz sampling with I²C error detection in ESP32 C++ firmware.
- Non-blocking microsecond-timestamped serial CSV protocol (`timestamp_us,ax,ay,az,gx,gy,gz`).
- PC-side high-throughput acquisition logger (`scripts/run_acquisition_check.py`).

#### ✅ Phase 3: Signal Processing & DSP (Features 10–21)
- Offline raw CSV loaders and stationary calibration offset estimation.
- 3-axis representation strategies: per-axis, Euclidean magnitude, and strongest tremor-axis selection.
- DC baseline / gravity removal and 4–12 Hz tremor / 0.5–4 Hz voluntary Butterworth bandpass filtering (`scipy.signal`).
- Overlapping windowing (1–2s, 50% overlap) and Welch PSD computation.
- Complete feature extraction: tremor-band power, total power, power ratio, spectral entropy, RMS amplitude, variance, and accelerometer/gyroscope magnitudes.
- Visual validation script (`scripts/run_pipeline_baseline.py`).

#### ✅ Phase 4: Tremor Detection & Machine Learning (Features 22–27)
- Subject-level dataset preparation using `GroupKFold` to prevent inter-subject window leakage.
- Non-ML signal-processing baseline threshold detector for benchmarking.
- Classical model training and evaluation suite (Logistic Regression, Linear/RBF SVM, Random Forest).
- ESP32-oriented model footprint evaluation and inference runtime module (`ml/inference.py`).

#### ✅ Phase 5: Frequency, Amplitude & Phase Estimation (Features 28–31)
- Peak-frequency extraction from Welch PSD with flat-spectrum fallback (`estimation/frequency_estimation.py`).
- Time-domain RMS amplitude extraction (`estimation/amplitude_estimation.py`).
- Experimental instantaneous phase-estimation module (`estimation/phase_estimation.py`, optional/disabled by default).

#### ✅ Phase 6: Stimulation Simulation (Features 32–35)
- Synthetic ground-truth tremor generator with controllable frequency, amplitude, and stochastic noise (`simulation/tremor_model.py`).
- 2nd-order ODE state-space stimulation response model with configurable physiological latency gating (`simulation/stimulation_model.py`).
- Tremor-band power reduction suppression metric calculator (`simulation/suppression.py`).
- Universal simulation facade interface `simulation.apply(params, tremor_state) -> SimResult` with defensive bounds checking.

#### ✅ Phase 7: Adaptive Closed-Loop Controller (Features 36–40)
- Confidence gating (`confidence >= 0.6`) and hysteresis-stabilized severity thresholding (`controller/decision_logic.py`).
- Initial stimulation parameter calculation (`controller/parameter_selection.py`) scaled to detected severity and frequency.
- Dynamic feedback adaptation (`controller/adaptation.py`): step-capped increase/decrease/maintain logic targeting desired suppression while minimizing stimulation exposure.
- Purely functional, deterministic state tracking (`controller/controller_state.py`).
- Complete single-cycle controller facade (`controller/cycle.py`).

### Test Suite & Health

- **230 automated unit, integration, and contract tests** passing.
- 1 test intentionally skipped pending real-device hardware session capture.

---

## Remaining Work & Research Roadmap

The remaining 20 features comprise closed-loop integration, comparative clinical/research validation, reporting dashboards, and on-device deployment profiling:

```
[Phase 1-7: Completed] ──► [Phase 8: Closed-Loop] ──► [Phase 9: Validation] ──► [Phase 10: Dashboards] ──► [Phase 11: On-Device] ──► [Phase N: Final]
                                (In Progress)
```

### Phase 8 — Closed-Loop Integration (Current)
*Connecting DSP, ML, Controller, and Simulation into a unified execution loop.*

- [ ] **Feature 41: Closed-Loop Runner (Single-Cycle, Synthetic Tremor)**
  - Implement `simulation/closed_loop_runner.py` wiring `extract_features -> ml.predict -> run_controller_cycle -> simulation.apply -> adapt_params`.
- [ ] **Feature 42: Multi-Cycle Closed-Loop Simulation**
  - Implement `scripts/run_closed_loop_simulation.py` to evaluate dynamic convergence over multi-window time series.
- [ ] **Feature 43: Closed-Loop Execution on Recorded Hardware Data**
  - Replay real captured wrist IMU datasets through the closed loop to verify tremor onset detection and suppression behavior on physiological signals.

### Phase 9 — Validation, Benchmarking & Robustness
*Quantitative comparative research demonstrating adaptive closed-loop efficacy over open-loop / fixed mitigation.*

- [ ] **Feature 44: No-Mitigation Baseline Experiment (`validation/experiments/no_mitigation.py`)**
  - Benchmark control run measuring baseline tremor trajectory.
- [ ] **Feature 45: Fixed-Parameter Mitigation Experiment (`validation/experiments/fixed_mitigation.py`)**
  - Non-adaptive static stimulation benchmark.
- [ ] **Feature 46: Adaptive Mitigation Experiment (`validation/experiments/adaptive_mitigation.py`)**
  - Full adaptive closed-loop benchmark.
- [ ] **Feature 47: Validation Metrics Module (`validation/metrics.py`)**
  - Calculate tremor power reduction (%), residual amplitude, time-to-target suppression, duty cycle, total electrical exposure, latency, and parameter oscillation/stability.
- [ ] **Feature 48: Voluntary-Movement Rejection Evaluation**
  - Quantify false-positive activation rates against 0.5–4 Hz intentional wrist movements.
- [ ] **Feature 49: Frequency and Amplitude Estimation Accuracy**
  - Quantify estimation error against known synthetic sweeps.
- [ ] **Feature 50: Phase Estimation Accuracy (Conditional)**
  - Quantify phase tracking error if Phase 5 experimental module is enabled.
- [ ] **Feature 51: System Robustness & Disturbance Testing**
  - Stress test closed-loop stability against IMU sensor noise, non-stationary frequency drift, sudden amplitude jumps, and low-confidence prediction corruption.

### Phase 10 — Visualization & Research Dashboards
*Visual inspection tools and figure generators for experimental reporting.*

- [ ] **Feature 52: Raw vs. Filtered Signal & PSD Visualizer (`visualization/signal_plots.py`)**
- [ ] **Feature 53: Detection & Severity Timeline Dashboard**
- [ ] **Feature 54: Adaptive Controller State & Parameter Trajectory Dashboard (`visualization/controller_dashboard.py`)**
- [ ] **Feature 55: Comparative Validation Report Generator (`visualization/validation_report.py`)**

### Phase 11 — Embedded / On-Device Deployment
*Profiling and preparing algorithms for eventual on-MCU execution.*

- [ ] **Feature 56: ESP32 Resource Profiling for Candidate ML Models**
  - Measure RAM, flash, and compute cycle budgets for model candidates.
- [ ] **Feature 57: On-Device Causal DSP Port (Conditional)**
  - Implement real-time causal bandpass filtering (`lfilter` / circular buffer) in C++.
- [ ] **Feature 58: On-Device Lightweight Inference Port (Conditional)**
  - C++ implementation of the selected lightweight classifier.
- [ ] **Feature 59: Real-Time Timing Validation**
  - Measure per-sample and per-window latency directly on the ESP32.

### Phase N — Final System Verification
- [ ] **Feature 60: Complete End-to-End System Verification**
  - Formal multi-subject validation run demonstrating all 10 project success criteria simultaneously.

---

## Repository Structure

```text
tremor-system/
├── firmware/                          # ESP32 C++ firmware (PlatformIO / Arduino Core)
│   ├── src/
│   │   ├── main.cpp                   # Main acquisition setup & scheduler
│   │   ├── mpu6050_driver.cpp/.h      # I²C sensor driver & DLPF config
│   │   ├── sampling_timer.cpp/.h      # 100 Hz hardware timer & ISR
│   │   └── serial_protocol.cpp/.h     # Microsecond-accurate CSV serialization
├── config/
│   └── system_config.yaml             # Central configuration (single source of truth)
├── tremor_system/
│   ├── config.py                      # Strongly-typed configuration loader dataclasses
│   └── types.py                       # Shared data contracts (StimParams, SimResult, etc.)
├── signal_processing/                 # DSP & Feature Engineering
│   ├── calibration.py                 # Stationary offset estimation & unit conversion
│   ├── axis_handling.py               # Per-axis, magnitude, and strongest-axis selection
│   ├── filtering.py                   # DC removal & Butterworth bandpass filters
│   ├── windowing.py                   # Overlapping window segmenter
│   ├── spectral_analysis.py           # Welch PSD, tremor-band power & spectral entropy
│   ├── time_domain.py                 # RMS amplitude and variance calculations
│   └── feature_extraction.py          # Unified feature vector assembler
├── ml/                                # Tremor Classification & Machine Learning
│   ├── dataset_builder.py             # Subject-level GroupKFold dataset constructor
│   ├── baseline_detector.py           # Signal-processing rule-based benchmark
│   ├── train.py                       # Model training & scaler persistence
│   ├── evaluate.py                    # Multi-metric evaluation (Precision/Recall/F1/FPR)
│   ├── model_selection.py             # Smallest-model selection logic
│   └── inference.py                   # Unified inference engine
├── estimation/                        # Signal Parameter Estimators
│   ├── frequency_estimation.py        # Dominant tremor-band peak frequency extractor
│   ├── amplitude_estimation.py        # Filtered tremor amplitude estimator
│   └── phase_estimation.py            # Experimental Hilbert phase tracking
├── controller/                        # Closed-Loop Adaptive Controller
│   ├── controller_state.py            # Immutable controller state container
│   ├── decision_logic.py              # Confidence gating & hysteresis decision engine
│   ├── parameter_selection.py         # Initial bounded stimulation parameter generator
│   ├── adaptation.py                  # Closed-loop parameter adjustment & exposure minimizer
│   └── cycle.py                       # Single-cycle controller facade
├── simulation/                        # Stimulation Simulation Subsystem
│   ├── tremor_model.py                # Synthetic ground-truth tremor generator
│   ├── stimulation_model.py           # 2nd-order ODE state-space response model
│   ├── suppression.py                 # Power reduction suppression calculator
│   ├── __init__.py                    # simulation.apply() unified facade
│   └── closed_loop_runner.py          # Multi-stage closed loop orchestrator (In Progress)
├── validation/                        # Comparative Benchmarking & Experiments
│   ├── experiments/                   # No-mitigation, fixed, and adaptive runners
│   ├── metrics.py                     # Research evaluation metrics
│   └── robustness_tests.py            # Noise & disturbance evaluation
├── visualization/                     # Research Visualizations & Dashboards
├── tests/                             # Comprehensive pytest test suite (230+ tests)
├── data/                              # Immutable raw sessions, features, and model artifacts
├── scripts/                           # CLI entry points and orchestration tools
├── context/                           # System architecture specifications & build plans
└── AGENTS.md                          # Engineering standards & context navigation rules
```

---

## Getting Started & Usage

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rahul-ez/tremor-trace.git
   cd tremor-trace
   ```

2. **Set up Python environment (Python 3.10+ required):**
   ```bash
   python -m venv venv
   # Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   # Linux / macOS:
   source venv/bin/activate

   pip install -r requirements.txt
   ```

### Hardware Telemetry & Data Logging

To record raw synchronized 100 Hz IMU telemetry from the wrist-worn ESP32 device:

```powershell
python scripts/run_acquisition_check.py `
  --subject-id subj01 `
  --session-id sess01 `
  --port COM9 `
  --baud 115200 `
  --duration 60
```

| Parameter | Default | Purpose |
|---|---:|---|
| `--subject-id` | `subj01` | Target subject identifier |
| `--session-id` | `sess01` | Recording session identifier |
| `--port` | Auto-detect | Serial COM port |
| `--baud` | `115200` | Serial transmission baud rate |
| `--duration` | Unlimited | Recording duration in seconds |
| `--overwrite` | Off | Overwrite existing session files |

### Signal Processing Baseline

To process a recorded session, apply Butterworth filtering, perform Welch PSD, and inspect raw vs. filtered waveforms:

```powershell
python scripts/run_pipeline_baseline.py --subject-id subj01 --session-id sess01
```

### Running Automated Tests

Run the full pytest suite:

```bash
pytest -v
```

Run focused tests for specific subsystems:
```bash
pytest tests/test_signal_processing.py
pytest tests/test_controller_cycle.py
pytest tests/test_simulation_apply.py
```

---

## Key Architectural Contracts & Invariants

1. **Strict Dependency Flow:** Data flows unidirectionally:
   $$\text{Firmware} \longrightarrow \text{DSP} \longrightarrow \text{ML / Estimation} \longrightarrow \text{Controller} \longrightarrow \text{Simulation} \longrightarrow \text{Validation}$$
   Downstream modules never access raw sensor signals or mutate upstream state directly.
2. **Subject-Level Split Integrity:** All ML evaluation uses `GroupKFold` by `subject_id` to strictly eliminate data leakage across overlapping windows.
3. **Bounded & Gated Control:** The controller never issues unbounded stimulation parameters. All changes are clamped by `max_delta_per_step` and gated by a minimum confidence score ($> 0.6$) and hysteresis band.
4. **Separation of Hysteresis & Tolerance:** Severity-based activation is gated by `hysteresis_pct`, while adaptation steady-state holding is governed by `suppression_tolerance_pct`.
5. **Deterministic Simulation:** All ODE integration (`scipy.integrate.solve_ivp`) enforces explicit `max_step` timing and reproducible pseudo-random seeds (`np.random.default_rng`).

---

## License & Research Disclaimer

This project is released under the MIT License for research and educational purposes.

```
DISCLAIMER: This software and associated hardware designs are for research and 
simulation purposes only. They are not medical devices, are not clinically validated, 
and must not be used for diagnosis, treatment, or direct automated electrical stimulation 
of human subjects.
```

