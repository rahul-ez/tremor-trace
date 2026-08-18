# Memory — Phases 1 & 2 Completed (Foundation & MPU6050 Acquisition)

Last updated: 2026-08-17 22:57 IST

## What was built

- **Feature 01**: Repository skeleton created (`firmware/`, `config/`, `data/{raw,processed,features,models,simulation,validation}/`, `signal_processing/`, `ml/`, `estimation/`, `controller/`, `simulation/`, `validation/experiments/`, `visualization/`, `tests/`, `scripts/`, `tremor_system/`), `.gitkeep` files, and `.gitignore`.
- **Feature 02**: `config/system_config.yaml` with explicit TBD annotations for open architecture questions, and `tremor_system/config.py` typed dataclass loader.
- **Feature 03**: `tremor_system/types.py` defining `StimParams`, `SimResult`, `InferenceResult`, `FeatureVector`, and `WindowResult` dataclasses.
- **Feature 04**: `pyproject.toml` with `pytest` configuration.
- **Feature 05**: `firmware/src/mpu6050_driver.h/.cpp` (I²C 400kHz, SDA=21, SCL=22, DLPF 42Hz, ±2g accel, ±250°/s gyro).
- **Feature 06**: `firmware/src/sampling_timer.h/.cpp` (100 Hz hardware timer ISR flag pattern).
- **Feature 07**: `firmware/src/serial_protocol.h/.cpp` (`timestamp_us,ax,ay,az,gx,gy,gz\n` @ 115200 baud).
- **Feature 08**: `firmware/src/main.cpp` and `firmware/platformio.ini` with `ERR,I2C_READ_FAILED` error handling.
- **Feature 09**: `scripts/run_acquisition_check.py` PC-side serial logger with stream validation and CSV output to `data/raw/<subject_id>/<session_id>/raw_stream.csv`.
- **Unit Tests**: 12 automated unit tests in `tests/test_config.py`, `tests/test_types.py`, and `tests/test_acquisition_logger.py` (100% passing).

## Decisions made

- Followed PlatformIO modular layout for `firmware/` (`src/main.cpp`, `mpu6050_driver`, `sampling_timer`, `serial_protocol`, `platformio.ini`) as required by `context/architecture.md`.
- Legacy `raw-signal-acquisition/` sketch directory identified as redundant and approved for safe deletion.
- All experimental/TBD configuration parameters are kept explicitly as `None` or commented TBDs in `system_config.yaml` and parsed cleanly by `load_config()`.

## Problems solved

- None; all features implemented cleanly per context specs.

## Current state

- Phase 1 (Foundation) and Phase 2 (MPU6050 Acquisition) are 100% complete and verified.
- 12 unit tests passing cleanly in `pytest`.
- `context/progress-tracker.md` updated to show Features 01–09 completed.

## Next session starts with

- **Phase 3 — Signal Processing**: Feature 10 (Raw Data Loader - Offline) in `signal_processing/calibration.py` to load `raw_stream.csv` into NumPy arrays with exact shape `(n_samples, 6)` and units intact.

## Open questions

- None blocking Phase 3.
