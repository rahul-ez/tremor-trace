# Project Overview

## About the Project

This project is a wearable, closed-loop hand-tremor detection and suppression system built around an **ESP32 and MPU6050 6-axis IMU**. The ESP32 acquires synchronized 3-axis accelerometer and 3-axis gyroscope data at a target sampling rate of **100 Hz** and streams the measurements over serial. **Serial Studio** can be used for real-time telemetry and visualization, while Python is used for offline signal processing, dataset preparation, ML development, simulation, and validation. The current implementation is simulation-first: no automatic electrical stimulation is applied to a person.

The signal-processing pipeline is designed to isolate pathological tremor from raw wrist motion. The planned flow is **data acquisition → sensor calibration → axis handling → baseline/gravity removal → band-pass filtering → overlapping windowing → tremor-band spectral analysis → time-domain amplitude analysis → feature extraction**. The baseline tremor-processing configuration uses a **4–12 Hz tremor band** and a **0.5–4 Hz voluntary-movement band**, with the exact filter boundaries treated as experimentally tunable. A Butterworth band-pass filter is the initial candidate. Windows of approximately **1–2 seconds with 50% overlap** are used, followed by **Welch PSD** analysis rather than relying only on a raw FFT. Core signal features include **tremor-band power, total power, tremor/total power ratio, dominant tremor frequency, RMS/amplitude, variance, spectral entropy, and accelerometer/gyroscope magnitude**. The system should initially compare individual-axis processing, 3-axis magnitude, and strongest-tremor-axis selection rather than permanently committing to one representation.
The ML stage follows a progression from a signal-processing baseline to lightweight classical machine-learning models. Candidate models are **Logistic Regression, SVM, and Random Forest**, with the smallest model meeting the project's detection requirements ultimately preferred for ESP32 deployment. The model produces **tremor/no-tremor classification, a severity estimate, and detection confidence**. Evaluation must use **subject-level train/validation/test splits or subject-level cross-validation** to prevent leakage from overlapping windows belonging to the same subject. Frequency is estimated from the dominant peak in the tremor-band PSD, while amplitude is obtained from the filtered tremor signal using RMS or a similar amplitude metric. Phase estimation is considered an advanced stage because filtering delay, noise, and non-stationary tremor make reliable phase estimation substantially harder.

The central system component is an **adaptive closed-loop controller**. The controller receives the ML confidence, tremor severity, amplitude, dominant frequency, and eventually phase information. It first determines whether mitigation is necessary, then selects simulated electrical-stimulation parameters such as amplitude/current, pulse frequency, pulse width, duty cycle, ON/OFF timing, and phase relationship. The simulated tremor response is measured after mitigation, and the controller compares the achieved suppression against a target. It then **increases, decreases, or maintains the stimulation parameters**, subject to bounded parameter ranges, confidence gating, hysteresis, and a capped adaptation rate. The optimization objective is to achieve the required tremor suppression while minimizing stimulation exposure and duty cycle. The complete loop is therefore:

**Detect → Analyze → Decide → Mitigate → Measure → Adapt → Repeat.**

The stimulation stage is currently entirely simulated using **Python ODE/state-space modelling or MATLAB/Simulink**. The simulated model represents the relationship between tremor characteristics and stimulation response for controller-development purposes; it is not treated as a clinically validated physiological model. Validation compares **no mitigation, fixed-parameter mitigation, and adaptive mitigation** using metrics such as tremor-band power reduction, percentage suppression, residual amplitude, time-to-target, duty cycle, total simulated exposure, false activation rate, latency, and controller stability.

### Current Implementation Architecture

```text
                    ┌────────────────────┐
                    │      MPU6050       │
                    │ 3-Axis Accel/Gyro   │
                    └─────────┬──────────┘
                              │ I²C
                              ▼
                    ┌────────────────────┐
                    │       ESP32        │
                    │                    │
                    │ 100 Hz acquisition │
                    │ Timestamping       │
                    │ Raw data streaming │
                    └─────────┬──────────┘
                              │ USB Serial
                    ┌─────────┴──────────┐
                    │                    │
                    ▼                    ▼
             ┌──────────────┐    ┌────────────────┐
             │ Serial Studio│    │ Python Pipeline│
             │ Live Telemetry│    │                │
             └──────────────┘    │ Calibration    │
                                 │ Filtering      │
                                 │ Windowing      │
                                 │ PSD / Features │
                                 └───────┬────────┘
                                         │
                                         ▼
                                 ┌────────────────┐
                                 │ Lightweight ML │
                                 │ Tremor / No    │
                                 │ Tremor         │
                                 │ Severity       │
                                 │ Confidence     │
                                 └───────┬────────┘
                                         │
                                         ▼
                                 ┌────────────────┐
                                 │ Frequency /    │
                                 │ Phase Analysis │
                                 └───────┬────────┘
                                         │
                                         ▼
                                 ┌────────────────┐
                                 │ Adaptive       │
                                 │ Controller     │
                                 └───────┬────────┘
                                         │
                                         ▼
                                 ┌────────────────┐
                                 │ Stimulation    │
                                 │ Simulation     │
                                 └───────┬────────┘
                                         │
                                         ▼
                                 ┌────────────────┐
                                 │ Suppression    │
                                 │ Measurement    │
                                 └───────┬────────┘
                                         │
                                         └──────► Adapt
```

### Signal-Processing Pipeline

The signal-processing implementation should be developed incrementally rather than implementing the entire system simultaneously:

```text
Raw MPU6050 data
       ↓
Sensor calibration
       ↓
Axis handling
       ↓
Gravity / baseline removal
       ↓
Band-pass filtering
       ↓
Overlapping windows
       ↓
Welch PSD
       ↓
Tremor-band analysis
       ├── Tremor-band power
       ├── Total power
       ├── Power ratio
       └── Dominant frequency
       ↓
Time-domain analysis
       └── RMS / amplitude
       ↓
Feature vector
       ↓
ML detection + severity
```

The first development milestone is to verify that the **raw 100 Hz ESP32 data is stable and physically sensible**. The second milestone is to implement the Python filtering and PSD pipeline and visually verify raw versus filtered signals and the tremor spectrum. Only after this baseline works reliably should ML and adaptive control be integrated.

### Hardware Acquisition

The current ESP32 firmware communicates with the MPU6050 over I²C using **SDA = GPIO 21 and SCL = GPIO 22**, with the I²C clock configured to **400 kHz**. The MPU6050 is configured for a nominal **100 Hz sampling rate**, ±2 g accelerometer range, ±250 °/s gyroscope range, and a DLPF configuration of `0x03`. The firmware reads the complete 14-byte MPU6050 measurement block and outputs:

```text
timestamp_us,ax,ay,az,gx,gy,gz
```

over serial at **115200 baud**.

### Technology Stack

```text
Embedded:
    ESP32
    MPU6050
    Arduino / ESP-IDF
    I²C
    100 Hz sensor acquisition

Telemetry / Visualization:
    Serial Studio
    USB Serial

Signal Processing:
    Python
    NumPy
    SciPy
    pandas
    Welch PSD
    Butterworth filtering

Machine Learning:
    scikit-learn
    Logistic Regression
    SVM
    Random Forest

Simulation:
    Python ODE / state-space model
    OR MATLAB / Simulink

Control:
    Rule-based adaptive closed-loop controller
    Confidence gating
    Thresholding
    Hysteresis
    Bounded parameter adaptation
    Minimum-exposure objective

Storage:
    CSV datasets
    No database currently

Authentication:
    None

Payments:
    None

Cloud Backend:
    None currently
```

### Development Philosophy

The project should be developed as a sequence of validated gates rather than as one large system. First establish reliable **ESP32 → MPU6050 → 100 Hz raw acquisition**, then validate the signal-processing pipeline, then establish a non-ML tremor detector, then compare lightweight ML models, then deploy the selected inference pipeline to the ESP32, and finally integrate the adaptive controller with the stimulation simulation. This prevents problems in the controller or ML layer from masking fundamental issues in sensor acquisition and signal processing.
---

## The Problem It Solves

The project addresses the problem of **continuously detecting and characterizing hand tremor and determining when and how much mitigation should be applied**. Raw wrist motion contains gravity, sensor bias, voluntary movement, posture changes, noise, and tremor simultaneously, making simple thresholding of raw accelerometer or gyroscope values unreliable. The system therefore needs to isolate the tremor-frequency component, distinguish tremor from voluntary movement, estimate tremor characteristics, and make mitigation decisions in real time.

The project specifically eliminates the need for a **fixed, manually selected mitigation strategy**. Instead of applying the same stimulation parameters regardless of the user's current tremor, the adaptive controller uses the detected tremor severity, frequency, confidence, and measured suppression to continuously adjust the simulated intervention. Its objective is a bounded feedback-control problem: **maximize tremor suppression while minimizing unnecessary stimulation exposure**, while avoiding unstable rapid ON/OFF switching through confidence gating, hysteresis, parameter bounds, and capped adaptation.

The current system is deliberately **simulation-first**. Electrical stimulation is not automatically applied to a human; instead, the controller interacts with a simulated tremor-response model. This allows the detection, decision, mitigation, measurement, and adaptation logic to be quantitatively validated before any future physical stimulation integration.

---

## Pages / Functional Views

The project is not currently a conventional SaaS/web application. Its interface is primarily a **real-time engineering and research monitoring system**, with Serial Studio providing live hardware telemetry and Python-based tools providing deeper analysis and simulation.

```text
/monitor
    Live ESP32/MPU6050 telemetry.
    Displays ax, ay, az, gx, gy, gz, timestamps,
    sampling information, and sensor status.

/signal-analysis
    Displays raw and calibrated signals, filtered
    tremor signals, individual-axis comparisons,
    tremor-band power, total power, RMS, and PSD.

/frequency-analysis
    Displays Welch PSD, tremor-band spectrum,
    dominant tremor frequency, spectral concentration,
    and frequency tracking over successive windows.

/detection
    Displays ML tremor/no-tremor classification,
    severity estimate, confidence, and detection history.

/controller
    Displays the adaptive controller state:
    detection confidence, severity, mitigation decision,
    target suppression, current suppression, and
    selected simulated stimulation parameters.

/simulation
    Displays the simulated tremor-response loop,
    including baseline tremor, stimulation waveform/model,
    residual tremor, suppression percentage, and
    controller adaptation.

/validation
    Displays experimental metrics including detection
    precision/recall/F1, frequency-estimation error,
    tremor-power reduction, residual amplitude,
    time-to-target, duty cycle, exposure,
    latency, false activation rate, and stability.
```

Serial Studio is primarily intended for **live ESP32 telemetry and hardware debugging**, while the signal-analysis, ML, controller, and stimulation-simulation views can be implemented in Python or another research visualization layer as the project progresses.

---

## Navigation

The project interface should follow the engineering pipeline rather than a traditional business-application navigation structure.

```text
MONITOR
   ↓
SIGNAL ANALYSIS
   ↓
FREQUENCY / FEATURE ANALYSIS
   ↓
TREMOR DETECTION
   ↓
ADAPTIVE CONTROLLER
   ↓
STIMULATION SIMULATION
   ↓
VALIDATION
```

### Monitor

Entry point for checking whether the ESP32 and MPU6050 are communicating correctly and whether the sensor stream is stable at approximately 100 Hz. Serial Studio is used here for live telemetry.

### Signal Analysis

Used to inspect the transformation from raw IMU data into a clean tremor signal. This includes calibration, baseline/gravity removal, filtering, windowing, and comparison of different axis-handling strategies.

### Frequency / Feature Analysis

Used to inspect the Welch PSD and extracted tremor features, particularly tremor-band power, total power, power ratio, RMS, and dominant frequency.

### Tremor Detection

Displays the output of the signal-processing baseline and lightweight ML models, including tremor/no-tremor classification, severity, and confidence.

### Adaptive Controller

Shows the controller's internal state and decision-making. The controller uses tremor characteristics and confidence to decide whether mitigation is required and adjusts bounded simulated stimulation parameters based on measured suppression.

### Stimulation Simulation

Runs the simulated closed-loop intervention. The simulated tremor response is fed back to the controller so that it can evaluate whether the target suppression level has been achieved and adapt accordingly.

### Validation

Provides quantitative comparison between no mitigation, fixed-parameter mitigation, and adaptive mitigation, including suppression, residual tremor, latency, duty cycle, simulated exposure, and controller stability.
---

## Core User Flow

### 1. Device Setup / Initialization

- The user connects the wrist-worn **ESP32 + MPU6050** device to the PC.
- The system initializes I²C communication and configures the MPU6050 for approximately **100 Hz sampling**, ±2 g accelerometer range, ±250 °/s gyroscope range, and DLPF configuration.
- Serial Studio can be opened to verify live telemetry and sensor stability.

### 2. Sensor Calibration

- The system collects stationary sensor measurements to estimate accelerometer and gyroscope offsets.
- Raw measurements are corrected using the estimated offsets.
- The user can verify that a stationary device produces stable acceleration and near-zero gyroscope readings.

### 3. Live Data Acquisition

- The ESP32 continuously reads:
    `ax, ay, az, gx, gy, gz`

- Each measurement is timestamped and streamed over USB serial.
- Serial Studio provides real-time visualization of the sensor signals.
- The raw stream can simultaneously be logged for later analysis.

### 4. Signal Processing

The acquired data is passed into the signal-processing pipeline:

```text
Raw IMU
  ↓
Calibration
  ↓
Axis handling
  ↓
Gravity / baseline removal
  ↓
Band-pass filtering
  ↓
Windowing
  ↓
Welch PSD
  ↓
Tremor feature extraction
```

The initial processing configuration uses a **4–12 Hz tremor band** and **0.5–4 Hz voluntary-movement band**. Approximately **1–2 second windows with 50% overlap** are used. A Butterworth filter is the initial candidate, while exact filter parameters remain experimentally tunable.

### 5. Tremor Analysis

For each analysis window, the system calculates:

- Tremor-band power
- Total signal power
- Tremor/total power ratio
- Dominant tremor frequency
- RMS / tremor amplitude
- Variance
- Spectral features where required
- Accelerometer/gyroscope magnitude

The system also evaluates different axis strategies during development:

- Individual-axis analysis
- 3-axis magnitude
- Strongest-tremor-axis selection

### 6. Tremor Detection and Severity Estimation

The extracted features are passed to the detection pipeline.

The development progression is:

```text
Signal-processing threshold baseline
              ↓
Feature engineering
              ↓
Logistic Regression / SVM / Random Forest
              ↓
Select smallest suitable model
              ↓
ESP32 deployment
```

The model produces:

- Tremor / no-tremor classification
- Tremor severity estimate
- Detection confidence

### 7. Frequency and Phase Estimation

When tremor is detected, the system estimates its dominant frequency from the tremor-band PSD.

Phase estimation is treated as a later-stage capability because filtering delay, noise, and non-stationary tremor can make phase estimation unreliable. The initial controller therefore prioritizes reliable detection, severity, amplitude, and frequency information.

### 8. Adaptive Controller

The controller receives the detection confidence and tremor characteristics and determines whether mitigation is required.

```text
Tremor detected?
      ↓
Check confidence
      ↓
Estimate severity
      ↓
Estimate frequency / phase
      ↓
Decide mitigation
      ↓
Select stimulation parameters
```

The controller uses bounded parameters, confidence gating, hysteresis, and a capped adaptation rate to prevent unstable rapid switching.

### 9. Stimulation Simulation

No automatic stimulation is currently applied to a person.

Instead, the controller drives a **simulated electrical-stimulation model**. The simulation models the effect of the selected stimulation parameters on the tremor signal.

### 10. Measure → Adapt

After simulated mitigation, the system measures the resulting tremor:

```text
Before mitigation
       ↓
Simulated stimulation
       ↓
After mitigation
       ↓
Measure suppression
       ↓
Compare with target
       ↓
Increase / decrease / maintain parameters
       ↓
Repeat
```

The controller's objective is to achieve the target tremor suppression using the **minimum necessary simulated stimulation exposure**.

### 11. Validation

The final system compares:

- No mitigation
- Fixed-parameter mitigation
- Adaptive mitigation

using suppression, residual tremor, time-to-target, duty cycle, exposure, latency, false activation rate, and stability metrics.
---

## Data Architecture

The project does not currently use a conventional user/database architecture. Data flows primarily between the **ESP32, serial telemetry layer, Python processing environment, ML models, and stimulation simulation**.

```text
                    ESP32
                      │
                      │ Raw IMU
                      ▼
               Serial / CSV Stream
                      │
             ┌────────┴────────┐
             ▼                 ▼
       Serial Studio        Python
       Visualization      Processing
                               │
                               ▼
                       Processed Windows
                               │
                               ▼
                         Feature Vector
                               │
                               ▼
                          ML Model
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
             Class          Severity       Confidence
                │              │              │
                └──────────────┼──────────────┘
                               ▼
                       Adaptive Controller
                               │
                               ▼
                    Stimulation Simulation
                               │
                               ▼
                     Suppression Metrics
                               │
                               └──────► Feedback
```

### Primary Data Entities

**Raw IMU samples**

```text
timestamp_us
ax
ay
az
gx
gy
gz
```

These originate from the MPU6050 and are acquired by the ESP32 at approximately 100 Hz.

**Processed signal windows**

Each window contains calibrated and filtered sensor data used for frequency-domain and time-domain analysis.

**Feature vectors**

Features include tremor-band power, total power, power ratio, dominant frequency, RMS/amplitude, variance, spectral features, and sensor magnitudes.

**ML outputs**

Each window can produce:

- Tremor/no-tremor label
- Severity
- Confidence

**Controller state**

The controller maintains the current mitigation decision, target suppression, simulated stimulation parameters, and adaptation state.

**Simulation data**

The stimulation simulation produces the resulting tremor signal and suppression measurements.

### Storage

The current implementation uses **CSV-based sensor recordings** and local files for experimentation, datasets, trained models, scalers, and validation results. There is currently **no database, authentication system, per-user cloud account, or payment system**.

For a future multi-user clinical/research system, subject/session IDs could be introduced, but that is outside the current implementation scope.

---

## Features In Scope

### Hardware & Acquisition

- ESP32 + MPU6050 integration
- 6-axis IMU acquisition
- 100 Hz target sampling
- Timestamped sensor measurements
- Accelerometer and gyroscope calibration
- Raw CSV/serial data streaming
- Serial Studio real-time telemetry

### Signal Processing

- Sensor offset calibration
- Gravity/baseline removal
- Individual-axis analysis
- 3-axis magnitude analysis
- Band-pass filtering
- 4–12 Hz tremor-band extraction
- 0.5–4 Hz voluntary-movement analysis
- 1–2 second overlapping windows
- Welch PSD
- Tremor-band power
- Total power
- Tremor/total power ratio
- Dominant tremor frequency
- Tremor RMS/amplitude
- Feature comparison across axis strategies

### ML Detection

- Signal-processing baseline detector
- Feature engineering
- Logistic Regression evaluation
- SVM evaluation
- Random Forest evaluation
- Tremor/no-tremor classification
- Severity estimation
- Detection confidence
- Subject-level validation
- Lightweight-model selection for ESP32

### Adaptive Control

- Tremor-confidence gating
- Mitigation threshold
- Severity-based decision making
- Frequency-aware parameter selection
- Bounded stimulation parameters
- Hysteresis
- Capped adaptation rate
- Target suppression
- Feedback-based parameter adjustment
- Minimum-exposure objective

### Simulation

- Simulated tremor model
- Simulated electrical-stimulation model
- Simulated tremor response
- Closed-loop Detect → Analyze → Decide → Mitigate → Measure → Adapt loop
- No-mitigation baseline
- Fixed-parameter mitigation
- Adaptive mitigation

### Validation

- Tremor detection metrics
- Frequency-estimation error
- Tremor-power reduction
- Residual tremor
- Time-to-target
- Duty cycle
- Simulated stimulation exposure
- Detection-to-actuation latency
- False activation rate
- Controller stability
- Noise and robustness testing

---

## Features Out of Scope

- Direct automatic electrical stimulation of a human during the current development phase
- Designing or manufacturing a custom TENS/EMS stimulator
- Clinical diagnosis of Parkinson's disease, essential tremor, or other neurological disorders
- Clinical claims regarding treatment effectiveness
- Clinically validated physiological stimulation-response modelling
- Large-scale cloud backend
- User accounts and authentication
- Patient-facing mobile application
- Payment/subscription infrastructure
- Medical-record management
- Remote clinician portal
- Real-time cloud analytics
- Reinforcement-learning-based controller
- Complex deep-learning architectures unless lightweight classical ML proves insufficient
- Full phase-locked stimulation as an initial controller capability
- Large-scale multi-user deployment
- Production medical-device certification
- Final human-subject trials during the simulation-first development phase

---

## Success Criteria

The project is considered technically successful when the following measurable outcomes are demonstrated:

1. **Reliable acquisition:** The ESP32 successfully acquires synchronized MPU6050 accelerometer and gyroscope data at approximately **100 Hz** without persistent I²C or timing failures.

2. **Valid signal processing:** The pipeline successfully transforms raw IMU data into a clean tremor-band signal and produces interpretable PSD, RMS, power, and dominant-frequency measurements.

3. **Tremor detection:** The baseline and selected ML model can distinguish tremor from voluntary movement on held-out data using subject-level validation, with precision, recall, F1, sensitivity, and false-positive rate reported rather than relying only on accuracy.

4. **Frequency estimation:** The system reliably estimates the dominant tremor frequency within an experimentally defined acceptable error relative to the known/generated ground truth.

5. **Real-time feasibility:** The selected signal-processing and ML pipeline meets the project's latency and memory constraints sufficiently to support eventual ESP32 deployment.

6. **Adaptive control:** The controller responds correctly to changing simulated tremor amplitude/frequency and adjusts the simulated stimulation parameters rather than continuously applying a fixed intervention.

7. **Suppression objective:** The adaptive controller achieves the defined target tremor suppression while using less simulated stimulation exposure/duty cycle than an unnecessarily aggressive fixed strategy.

8. **Closed-loop operation:** The complete pipeline operates end-to-end:

```text
Sense
  ↓
Process
  ↓
Detect
  ↓
Estimate
  ↓
Decide
  ↓
Mitigate
  ↓
Measure
  ↓
Adapt
  ↺
```

9. **Robustness:** The system remains stable under simulated sensor noise, varying tremor frequency/amplitude, voluntary movement, filtering delay, and intentionally incorrect/low-confidence detections.

10. **Reproducible validation:** Results are quantitatively documented for **no mitigation vs fixed mitigation vs adaptive mitigation**, including tremor-power reduction, residual amplitude, time-to-target, duty cycle, simulated exposure, latency, and controller stability.
