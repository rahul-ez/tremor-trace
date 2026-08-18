# Progress Tracker

Update this file after every completed feature.

---

## Current Status

**Phase:** Phase 3 — Signal Processing
**Last completed:** 12 Axis Handling Strategies
**Next:** 13 Gravity / Baseline Removal
**Status:** In progress (Phase 3 partial, 3/21 features complete)

---

## Progress

**Phase 1 — Foundation**
- [x] 01 Repository and Folder Structure
- [x] 02 Central Configuration File and Loader
- [x] 03 Shared Data-Contract Types
- [x] 04 Test and Verification Infrastructure

**Phase 2 — MPU6050 Acquisition**
- [x] 05 MPU6050 I²C Driver
- [x] 06 Timed 100 Hz Acquisition Loop
- [x] 07 Serial CSV Output Protocol
- [x] 08 I²C Failure Handling
- [x] 09 PC-Side Raw Data Logger

**Phase 3 — Signal Processing**
- [x] 10 Raw Data Loader (Offline)
- [x] 11 Sensor Calibration (Offset Estimation)
- [x] 12 Axis Handling Strategies
- [ ] 13 Gravity / Baseline Removal
- [ ] 14 Band-Pass Filtering (Tremor + Voluntary Bands)
- [ ] 15 Complete Feature 12 — Strongest-Axis Selection
- [ ] 16 Overlapping Windowing
- [ ] 17 Welch PSD Computation
- [ ] 18 Tremor-Band Power, Total Power, Power Ratio, Dominant Frequency
- [ ] 19 Time-Domain Features (RMS, Variance)
- [ ] 20 Feature Vector Assembly
- [ ] 21 Milestone 2 Verification Script — Raw vs. Filtered Visual Check

**Phase 4 — Tremor Detection and ML**
- [ ] 22 Dataset Builder with Subject-Level Splitting
- [ ] 23 Baseline Signal-Processing Threshold Detector
- [ ] 24 Candidate ML Model Training
- [ ] 25 Model Evaluation and Comparison
- [ ] 26 Smallest-Model Selection for ESP32
- [ ] 27 ML Inference Module

**Phase 5 — Frequency, Amplitude, and Phase Estimation**
- [ ] 28 Dominant Frequency Estimation Module
- [ ] 29 Amplitude Estimation Module
- [ ] 30 Frequency/Amplitude Integration into ML Inference Output
- [ ] 31 Phase Estimation (Advanced, Experimental)

**Phase 6 — Stimulation Simulation**
- [ ] 32 Synthetic Tremor Signal Generator
- [ ] 33 Simulated Stimulation-Response Model
- [ ] 34 Suppression Measurement
- [ ] 35 Simulation Interface — `simulation.apply()`

**Phase 7 — Adaptive Controller**
- [ ] 36 Confidence Gating and Mitigation Decision Logic
- [ ] 37 Bounded Stimulation Parameter Selection
- [ ] 38 Adaptation Logic (Increase/Decrease/Maintain)
- [ ] 39 Controller State Management
- [ ] 40 Full Controller Loop (Detect → Decide → Mitigate, Single Cycle)

**Phase 8 — Closed-Loop Integration**
- [ ] 41 Closed-Loop Runner (Single-Cycle, Simulated Tremor Input)
- [ ] 42 Multi-Cycle Closed-Loop Simulation
- [ ] 43 Closed-Loop with Recorded Hardware Data

**Phase 9 — Validation and Robustness**
- [ ] 44 No-Mitigation Baseline Experiment
- [ ] 45 Fixed-Parameter Mitigation Experiment
- [ ] 46 Adaptive Mitigation Experiment
- [ ] 47 Validation Metrics Module
- [ ] 48 Detection Performance and Voluntary-Movement Rejection
- [ ] 49 Frequency and Amplitude Estimation Accuracy
- [ ] 50 Phase Accuracy (Conditional — Only If Feature 31 Enabled)
- [ ] 51 Robustness Testing

**Phase 10 — Visualization and Monitoring**
- [ ] 52 Raw and Filtered Signal Plots
- [ ] 53 Detection and Severity Dashboard Plot
- [ ] 54 Controller Dashboard
- [ ] 55 Validation Report Visualization

**Phase 11 — ESP32 Real-Time Deployment**
- [ ] 56 ESP32 Resource Profiling of Selected Model
- [ ] 57 On-Device Filtering/Windowing Port (Conditional)
- [ ] 58 On-Device Inference Port (Conditional)
- [ ] 59 Full Real-Time On-Device Acquisition + Processing Timing Validation

**Phase N — Final Integration**
- [ ] 60 Complete End-to-End System Verification

---

## Decisions Made During Build

(None yet.)

---

## Notes

(None yet.)
