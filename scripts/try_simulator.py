import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add the project root to sys.path so all packages are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulation import apply
from simulation.tremor_model import generate_synthetic_tremor
from tremor_system.config import load_config
from tremor_system.types import StimParams

cfg = load_config()

# --- 1. Create a synthetic 6 Hz tremor signal (2 seconds, 100 Hz) ---
freq_hz     = 6.0       # tremor frequency in the 4–12 Hz band
amplitude   = 0.4       # peak amplitude in g
duration_s  = 10.0
sample_rate = cfg.sensor.sample_rate_hz   # 100 Hz from config

signal = generate_synthetic_tremor(freq_hz, amplitude, duration_s, sample_rate)

# --- 2. Build the tremor_state dict the simulation expects ---
tremor_state = {
    "y0":           np.array([signal[0], (signal[1] - signal[0]) * sample_rate]),
    "duration_s":   duration_s,
    "timestep_s":   cfg.simulation.timestep_s,   # 0.001 s
    "signal":       signal,
    "sample_rate_hz": sample_rate,
}

# --- 3. Choose stimulation parameters (try changing amplitude!) ---
params = StimParams(
    amplitude         = 100,     # ← turn this up/down to see more/less suppression
    pulse_frequency_hz= 50.0,
    pulse_width_us    = 200.0,
    duty_cycle        = 0.2,
    on_off_timing     = (100.0, 50.0),
)

# --- 4. Run the simulation ---
result = apply(params, tremor_state)

print(f"Suppression:        {result.achieved_suppression_pct:.1f} %")
print(f"Residual amplitude: {result.residual_amplitude:.4f} g  (RMS)")
print(f"Latency modelled:   {result.latency_ms:.1f} ms")
print(f"Stability warning:  {result.stability_warning}")

# --- 5. Plot pre vs post ---
time_s = np.arange(signal.size) / sample_rate
plt.figure(figsize=(10, 4))
plt.plot(time_s, signal,                      label="Pre-mitigation")
plt.plot(time_s, result.post_mitigation_signal, label="Post-mitigation", alpha=0.8)
plt.axvline(cfg.simulation.latency_ms / 1000, color="r", linestyle="--", label="Latency gate")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (g)")
plt.title(f"Tremor Simulation — {result.achieved_suppression_pct:.1f}% suppression")
plt.legend()
plt.tight_layout()
plt.show()