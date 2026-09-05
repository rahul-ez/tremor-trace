# Progress Tracker

Update this file after every completed feature.

---

## Current Status

**Phase:** Phase 11 — ESP32 Real-Time Deployment
**Last completed:** 55 Validation Report Visualization — Phase 10 complete. Additionally: Live Demo Dashboard (`scripts/run_live_demo.py`) built and smoke-tested.
**Next:** 56 ESP32 Resource Profiling of Selected Model
**Status:** Phase 10 complete. Live real-time demo dashboard added as a demonstration-only script (outside the numbered build plan). Full suite: 275 passed, 8 skipped (unchanged — no production modules modified).

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
- [x] 13 Gravity / Baseline Removal
- [x] 14 Band-Pass Filtering (Tremor + Voluntary Bands)
- [x] 15 Complete Feature 12 — Strongest-Axis Selection
- [x] 16 Overlapping Windowing
- [x] 17 Welch PSD Computation
- [x] 18 Tremor-Band Power, Total Power, Power Ratio, Dominant Frequency
- [x] 19 Time-Domain Features (RMS, Variance)
- [x] 20 Feature Vector Assembly (including magnitude features)
- [x] 21 Milestone 2 Verification Script — Raw vs. Filtered Visual Check

**Phase 4 — Tremor Detection and ML**
- [x] 22 Dataset Builder with Subject-Level Splitting
- [x] 23 Baseline Signal-Processing Threshold Detector
- [x] 24 Candidate ML Model Training
- [x] 25 Model Evaluation and Comparison
- [x] 26 Smallest-Model Selection for ESP32
- [x] 27 ML Inference Module

**Phase 5 — Frequency, Amplitude, and Phase Estimation**
- [x] 28 Dominant Frequency Estimation Module
- [x] 29 Amplitude Estimation Module
- [x] 30 Frequency/Amplitude Integration into ML Inference Output
- [x] 31 Phase Estimation (Advanced, Experimental)

**Phase 6 — Stimulation Simulation**
- [x] 32 Synthetic Tremor Signal Generator
- [x] 33 Simulated Stimulation-Response Model
- [x] 34 Suppression Measurement
- [x] 35 Simulation Interface — `simulation.apply()`

**Phase 7 — Adaptive Controller**
- [x] 36 Confidence Gating and Mitigation Decision Logic
- [x] 37 Bounded Stimulation Parameter Selection
- [x] 38 Adaptation Logic (Increase/Decrease/Maintain)
- [x] 39 Controller State Management
- [x] 40 Full Controller Loop (Detect → Decide → Mitigate, Single Cycle)

**Phase 8 — Closed-Loop Integration**
- [x] 41 Closed-Loop Runner (Single-Cycle, Simulated Tremor Input)
- [x] 42 Multi-Cycle Closed-Loop Simulation
- [x] 43 Closed-Loop with Recorded Hardware Data

**Phase 9 — Validation and Robustness**
- [x] 44 No-Mitigation Baseline Experiment
- [x] 45 Fixed-Parameter Mitigation Experiment
- [x] 46 Adaptive Mitigation Experiment
- [x] 47 Validation Metrics Module
- [x] 48 Detection Performance and Voluntary-Movement Rejection
- [x] 49 Frequency and Amplitude Estimation Accuracy
- [x] 50 Phase Accuracy (Conditional — Only If Feature 31 Enabled) -- verified skip-path since disabled
- [x] 51 Robustness Testing

**Phase 10 — Visualization and Monitoring**
- [x] 52 Raw and Filtered Signal Plots
- [x] 53 Detection and Severity Dashboard Plot
- [x] 54 Controller Dashboard
- [x] 55 Validation Report Visualization

**Phase 11 — ESP32 Real-Time Deployment**
- [ ] 56 ESP32 Resource Profiling of Selected Model
- [ ] 57 On-Device Filtering/Windowing Port (Conditional)
- [ ] 58 On-Device Inference Port (Conditional)
- [ ] 59 Full Real-Time On-Device Acquisition + Processing Timing Validation

**Phase N — Final Integration**
- [ ] 60 Complete End-to-End System Verification

---

## Decisions Made During Build

(None yet — decisions for the numbered build plan features will be recorded here as Phase 11 progresses.)

**Live Demo Dashboard decisions (outside numbered build plan):**
- Rendering: Plotly Dash chosen over matplotlib animation — browser-based, no GUI toolkit required, cleaner demo display.
- Two update tiers: raw/filtered signal panels at 200 ms; analysis panels (PSD, ML, controller, suppression, before/after) at 800 ms.
- Threading: reader (daemon) + processor (daemon) threads; Dash Flask server on main thread; SharedState under threading.Lock.
- Calibration: 3-second stationary hold at startup; status banner shows countdown; active processing begins only after offsets are estimated.
- Replay mode: ReplaySource streams pre-recorded CSV with time.sleep(1/rate_hz) pacing; banner clearly shows "REPLAY -- pre-recorded data" in amber vs "LIVE DATA" in green.
- Suppression display: "Before" panel = real tremor signal from sensor; "After" panel = ODE-computed simulation.apply() output; both labelled explicitly; annotation reads "model-predicted reduction, not a physical measurement".
- New dependency: dash (+ transitive plotly, flask) added to requirements.txt and library-docs.md; used in scripts/ only.


---

## Notes

(None yet.)