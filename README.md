# Tremor Trace

Building a glove using ESP32 and MPU6050 that detects hand tremors and simulate tremor suppressions using adaptive closed loop FES system.

### Notes:
Use the following parameters while running scripts/run_acquisition_check.py:
You can pass these command-line parameters:

| Parameter | Default | Purpose |
|---|---:|---|
| `--subject-id` | `subj01` | Subject folder name |
| `--session-id` | `sess01` | Session folder name |
| `--port` | Auto-detect | Serial port, e.g. `COM9` |
| `--baud` | `115200` | Serial baud rate |
| `--duration` | Unlimited | Recording duration in seconds |
| `--overwrite` | Off | Replace the existing CSV instead of appending |

Example for a new session:

```powershell
python scripts/run_acquisition_check.py `
  --subject-id subj01 `
  --session-id sess02 `
  --port COM9 `
  --baud 115200 `
  --duration 60
```

To intentionally replace an existing session:

```powershell
python scripts/run_acquisition_check.py --session-id sess01 --port COM9 --overwrite
```

You can also view them with:

```powershell
python scripts/run_acquisition_check.py --help
```
