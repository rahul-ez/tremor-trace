"""Real-time tremor detection and adaptive control demonstration dashboard.

Reads from a live ESP32/MPU6050 serial stream (--port) or replays a
pre-recorded CSV at real-time pacing (--replay), runs the complete pipeline
on each incoming window, and displays results in a Dash browser dashboard
that updates live.

Usage -- live hardware:
    python scripts/run_live_demo.py --port COM5

Usage -- pre-recorded replay:
    python scripts/run_live_demo.py --replay data/raw/subj01/sess01/raw_stream.csv

The dashboard opens at http://127.0.0.1:8050

Pipeline per window cycle:
    calibrated samples -> bandpass filter -> feature extraction
    -> ML inference -> controller cycle -> simulation.apply() [if mitigating]
    -> adapt_params() -> SharedState update -> Dash callback reads SharedState

Architecture boundaries respected:
- All signal processing, ML, controller, and simulation logic is in the
  existing production modules.  This script is an orchestration entry point
  only (scripts/ role per architecture.md).
- No Dash callback mutates controller or simulation state; all mutations
  happen in the processor thread.  Callbacks are read-only observers.
- SharedState is passed explicitly to threads; no module-level mutable
  globals except the single _state reference required by Dash callbacks.
- Stimulation is and remains simulation-only; no physical actuation path
  exists anywhere in this script.
"""

from __future__ import annotations

import argparse
import collections
import csv
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional
import sys

# Ensure the repository root is on the module search path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from numpy.typing import NDArray
from dash import Dash, Input, Output, dcc, html
import plotly.graph_objects as go

from controller.adaptation import adapt_params
from controller.controller_state import ControllerState
from controller.cycle import run_controller_cycle
from ml.inference import predict
from signal_processing.calibration import apply_calibration, estimate_offsets
from signal_processing.feature_extraction import extract_features
from signal_processing.filtering import bandpass_filter, remove_baseline
from signal_processing.spectral_analysis import compute_welch_psd
from simulation import apply as apply_simulation
from simulation.closed_loop_runner import SIMULATION_SESSION_ID, SIMULATION_SUBJECT_ID
from tremor_system.config import Config, load_config
from tremor_system.types import InferenceResult, SimResult, StimParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CALIBRATION_SECONDS: int = 3
DISPLAY_WINDOW_S: float = 5.0       # rolling display buffer: last 5 seconds of raw/filtered signal
MAX_CYCLE_HISTORY: int = 60         # number of past cycles shown in time-series panels
FAST_INTERVAL_MS: int = 200         # raw + filtered signal refresh (ms)
SLOW_INTERVAL_MS: int = 800         # analysis panels refresh (ms)

DEFAULT_MODEL_PATH: Path = PROJECT_ROOT / "data" / "models" / "model_random_forest_v1.pkl"
DEFAULT_SCALER_PATH: Path = PROJECT_ROOT / "data" / "models" / "scaler_v1.pkl"

STATUS_CALIBRATING: str = "CALIBRATING"
STATUS_ACTIVE: str = "ACTIVE"
STATUS_DISCONNECTED: str = "DISCONNECTED"

SOURCE_LIVE: str = "LIVE"
SOURCE_REPLAY: str = "REPLAY"

# ---------------------------------------------------------------------------
# Color palette (all inline style usage must reference this dict)
# ---------------------------------------------------------------------------

_C: dict[str, str] = {
    "bg":             "#0d1117",
    "surface":        "#161b22",
    "border":         "#30363d",
    "text":           "#e6edf3",
    "text_muted":     "#8b949e",
    "accent":         "#58a6ff",
    "green":          "#3fb950",
    "red":            "#f85149",
    "amber":          "#d29922",
    "live_border":    "#3fb950",
    "replay_border":  "#d29922",
    "disc_border":    "#f85149",
}

# ---------------------------------------------------------------------------
# SharedState
# ---------------------------------------------------------------------------


class SharedState:
    """Thread-safe container for all state shared between threads and Dash callbacks.

    All mutable fields must be read and written under self.lock.
    config and source_type/source_label are set once at construction and are
    safe to read from callbacks without acquiring the lock.
    """

    def __init__(
        self,
        source_type: str,
        source_label: str,
        config: Config,
    ) -> None:
        self.lock: threading.Lock = threading.Lock()

        # Immutable after construction — safe to read without lock.
        self.source_type: str = source_type
        self.source_label: str = source_label
        self.config: Config = config

        # Pipeline status (mutable, protected by lock).
        self.status: str = STATUS_CALIBRATING
        self.calibration_countdown: int = CALIBRATION_SECONDS

        # Rolling display buffers — per-sample, updated by processor thread.
        # Sized in _processor_thread_fn once sample_rate_hz is known.
        self.raw_deque: collections.deque = collections.deque(maxlen=500)
        self.filtered_deque: collections.deque = collections.deque(maxlen=500)
        self.time_deque: collections.deque = collections.deque(maxlen=500)

        # Latest window analysis results — updated per window by processor thread.
        self.latest_freqs: Optional[NDArray] = None
        self.latest_psd: Optional[NDArray] = None
        self.latest_inference: Optional[InferenceResult] = None
        self.latest_stim_params: Optional[StimParams] = None
        self.latest_sim_result: Optional[SimResult] = None

        # Latest before/after waveform for the most recent window.
        self.before_window: Optional[NDArray] = None
        self.after_window: Optional[NDArray] = None

        # Time-series history for the controller / suppression charts.
        self.amplitude_history: list[float] = []
        self.suppression_history: list[float] = []
        self.cycle_count: int = 0

        # Controller state persists across cycles.
        self.controller_state: ControllerState = ControllerState()


# Module-level reference — set in main() before app.run().
# Standard Dash single-worker-process shared-state pattern (see library-docs.md).
_state: Optional[SharedState] = None

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------


class SerialSource:
    """Reads live samples from an ESP32 serial port.

    Yields raw sample dicts: {timestamp_us, ax, ay, az, gx, gy, gz}.
    Iteration stops on serial error or disconnect.
    """

    def __init__(self, port: str, baud: int = 115200) -> None:
        try:
            import serial
        except ImportError as exc:
            raise ImportError(
                "pyserial is required for live serial mode. "
                "It is already listed in requirements.txt; run: pip install pyserial"
            ) from exc
        self._ser = serial.Serial(port, baud, timeout=2.0)
        logger.info("Serial port opened: %s at %d baud.", port, baud)

    def __iter__(self):
        while True:
            try:
                raw_bytes = self._ser.readline()
                line = raw_bytes.decode("ascii", errors="ignore").strip()
            except Exception as exc:
                logger.error("Serial read error: %s", exc)
                return

            if not line or line.upper().startswith("ERR"):
                continue

            parts = line.split(",")
            if len(parts) != 7:
                continue

            try:
                yield {
                    "timestamp_us": int(parts[0]),
                    "ax": int(parts[1]),
                    "ay": int(parts[2]),
                    "az": int(parts[3]),
                    "gx": int(parts[4]),
                    "gy": int(parts[5]),
                    "gz": int(parts[6]),
                }
            except ValueError:
                continue


class ReplaySource:
    """Replays a pre-recorded raw_stream.csv at real-time pacing.

    Sleeps between samples to match the configured sensor sample rate so the
    pipeline receives data at the same rate as live acquisition.  The
    dashboard clearly shows SOURCE: REPLAY to distinguish this from live data.
    """

    def __init__(self, path: Path, rate_hz: float = 100.0) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Replay file not found: {path}")
        self._path: Path = path
        self._interval_s: float = 1.0 / rate_hz
        logger.info("Replay source: %s at %.1f Hz.", path, rate_hz)

    def __iter__(self):
        with open(self._path, newline="", encoding="ascii", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                # Skip header rows (non-numeric first field).
                first = row[0].strip().lstrip("+-")
                if not first.isdigit():
                    continue
                if len(row) < 7:
                    continue
                try:
                    yield {
                        "timestamp_us": int(row[0]),
                        "ax": int(row[1]),
                        "ay": int(row[2]),
                        "az": int(row[3]),
                        "gx": int(row[4]),
                        "gy": int(row[5]),
                        "gz": int(row[6]),
                    }
                    time.sleep(self._interval_s)
                except ValueError:
                    continue

# ---------------------------------------------------------------------------
# Reader thread
# ---------------------------------------------------------------------------


def _reader_thread_fn(
    source,
    sample_queue: queue.Queue,
    state: SharedState,
) -> None:
    """Read samples from source and enqueue them for the processor thread."""
    try:
        for sample in source:
            sample_queue.put(sample)
    except Exception as exc:
        logger.error("Reader thread error: %s", exc)
    finally:
        with state.lock:
            if state.status == STATUS_ACTIVE:
                state.status = STATUS_DISCONNECTED
        logger.warning("Reader thread exited — source exhausted or disconnected.")

# ---------------------------------------------------------------------------
# Processor thread
# ---------------------------------------------------------------------------


def _processor_thread_fn(
    sample_queue: queue.Queue,
    state: SharedState,
    config: Config,
    model_path: Path,
    scaler_path: Path,
) -> None:
    """Accumulate samples, calibrate, run full pipeline per window, update SharedState."""

    sample_rate_hz: float = config.sensor.sample_rate_hz
    window_samples: int = int(config.signal.window_length_s * sample_rate_hz)
    overlap_pct: float = config.signal.window_overlap_pct
    step_samples: int = max(1, int(window_samples * (1.0 - overlap_pct / 100.0)))
    calib_needed: int = int(CALIBRATION_SECONDS * sample_rate_hz)
    tremor_band: tuple[float, float] = tuple(config.signal.tremor_band_hz)  # type: ignore[assignment]
    filter_order: int = config.signal.filter_order
    display_maxlen: int = int(DISPLAY_WINDOW_S * sample_rate_hz)

    # Resize display deques now that sample_rate_hz is known.
    with state.lock:
        state.raw_deque = collections.deque(maxlen=display_maxlen)
        state.filtered_deque = collections.deque(maxlen=display_maxlen)
        state.time_deque = collections.deque(maxlen=display_maxlen)

    # Calibration accumulators.
    calib_raw: list[NDArray] = []
    offsets: Optional[NDArray] = None

    # Real-time windowing: rolling deques of exactly window_samples, filled sample-by-sample.
    accel_buf: collections.deque = collections.deque(maxlen=window_samples)  # (3,) per entry
    gyro_buf: collections.deque = collections.deque(maxlen=window_samples)   # (3,) per entry

    samples_since_last_process: int = 0
    sample_count: int = 0
    cycle_count: int = 0

    while True:
        # Block briefly; loop back to check for disconnect if nothing arrives.
        try:
            sample = sample_queue.get(timeout=1.0)
        except queue.Empty:
            with state.lock:
                if state.status == STATUS_DISCONNECTED:
                    break
            continue

        raw_lsb = np.array(
            [
                sample["ax"], sample["ay"], sample["az"],
                sample["gx"], sample["gy"], sample["gz"],
            ],
            dtype=np.float64,
        )
        sample_count += 1
        t_sample: float = sample_count / sample_rate_hz

        # ----------------------------------------------------------------
        # Phase 1: Calibration — collect stationary samples, estimate offsets.
        # ----------------------------------------------------------------
        if offsets is None:
            calib_raw.append(raw_lsb)
            elapsed_samples: int = len(calib_raw)
            countdown: int = max(0, CALIBRATION_SECONDS - int(elapsed_samples / sample_rate_hz))
            with state.lock:
                state.calibration_countdown = countdown

            if elapsed_samples >= calib_needed:
                calib_array = np.array(calib_raw, dtype=np.float64)  # (n, 6)
                offsets = estimate_offsets(calib_array)
                with state.lock:
                    state.status = STATUS_ACTIVE
                logger.info("Calibration complete. Offsets: %s", offsets)
            continue

        # ----------------------------------------------------------------
        # Phase 2: Active — calibrate sample, update display buffer, window.
        # ----------------------------------------------------------------
        calibrated = apply_calibration(raw_lsb.reshape(1, 6), offsets, config)[0]  # (6,)
        accel_cal: NDArray = calibrated[:3]   # g, three axes
        gyro_cal: NDArray = calibrated[3:]    # deg/s, three axes

        accel_buf.append(accel_cal)
        gyro_buf.append(gyro_cal)

        # Accel magnitude used as the raw display signal (no axis selection required per sample).
        accel_mag: float = float(np.linalg.norm(accel_cal))
        with state.lock:
            state.raw_deque.append(accel_mag)
            state.time_deque.append(t_sample)

        # Cannot process a window until the rolling buffer is full.
        if len(accel_buf) < window_samples:
            continue

        samples_since_last_process += 1
        if samples_since_last_process < step_samples:
            continue
        samples_since_last_process = 0

        # Snapshot the current window.
        accel_win: NDArray = np.array(list(accel_buf), dtype=np.float64)   # (window_samples, 3)
        gyro_win: NDArray = np.array(list(gyro_buf), dtype=np.float64)     # (window_samples, 3)

        # Accel magnitude as the single-axis analysis signal.
        analysis_signal: NDArray = np.linalg.norm(accel_win, axis=1)       # (window_samples,)

        # Signal processing: baseline removal then tremor-band bandpass filter.
        baseline_removed: NDArray = remove_baseline(analysis_signal)
        filtered: NDArray = bandpass_filter(
            baseline_removed, sample_rate_hz, tremor_band, filter_order
        )

        # Append the freshest step_samples to the filtered display buffer.
        with state.lock:
            for v in filtered[-step_samples:]:
                state.filtered_deque.append(float(v))

        # Feature extraction.
        nperseg: int = filtered.size
        try:
            feature_vec = extract_features(
                filtered,
                sample_rate_hz,
                accel_win,
                gyro_win,
                subject_id=SIMULATION_SUBJECT_ID,
                session_id=SIMULATION_SESSION_ID,
                window_id=cycle_count,
                nperseg=nperseg,
            )
        except Exception as exc:
            logger.warning("Feature extraction failed (cycle %d): %s", cycle_count, exc)
            continue

        # ML inference.
        try:
            ml_output: InferenceResult = predict(
                feature_vec.to_dict(),
                filtered,
                sample_rate_hz,
                model_path,
                scaler_path,
                nperseg=nperseg,
                config=config,
            )
        except Exception as exc:
            logger.warning("ML inference failed (cycle %d): %s", cycle_count, exc)
            continue

        # Controller cycle — reads and updates controller state explicitly.
        with state.lock:
            ctrl_state = state.controller_state
        stim_params, new_ctrl_state = run_controller_cycle(ml_output, ctrl_state, config)

        # Simulation — only when the controller decided to mitigate.
        before_win: NDArray = filtered.copy()
        sim_result: Optional[SimResult] = None
        new_params: Optional[StimParams] = stim_params

        if stim_params is not None:
            y0 = np.array([filtered[0], 0.0], dtype=np.float64)
            tremor_state_dict: dict = {
                "y0": y0,
                "duration_s": float(nperseg) / sample_rate_hz,
                "timestep_s": config.simulation.timestep_s,
                "signal": filtered,
                "sample_rate_hz": sample_rate_hz,
            }
            try:
                sim_result = apply_simulation(stim_params, tremor_state_dict, config)
                new_params, new_ctrl_state = adapt_params(
                    stim_params,
                    sim_result.achieved_suppression_pct,
                    config,
                    new_ctrl_state,
                    sim_result.stability_warning,
                )
            except Exception as exc:
                logger.warning("Simulation/adaptation failed (cycle %d): %s", cycle_count, exc)
                sim_result = None
                new_params = stim_params

        after_win: NDArray = (
            sim_result.post_mitigation_signal.copy()
            if sim_result is not None
            else before_win.copy()
        )

        # Welch PSD for the frequency-domain panel.
        try:
            freqs, psd = compute_welch_psd(filtered, sample_rate_hz, nperseg)
        except Exception:
            freqs, psd = None, None

        cycle_count += 1

        # Write all results to shared state under a single lock acquisition.
        amplitude: float = new_params.amplitude if new_params is not None else 0.0
        suppression: float = (
            sim_result.achieved_suppression_pct if sim_result is not None else 0.0
        )

        with state.lock:
            state.controller_state = new_ctrl_state
            state.latest_inference = ml_output
            state.latest_stim_params = new_params
            state.latest_sim_result = sim_result
            state.latest_freqs = freqs
            state.latest_psd = psd
            state.before_window = before_win
            state.after_window = after_win
            state.cycle_count = cycle_count
            state.amplitude_history.append(amplitude)
            state.suppression_history.append(suppression)
            if len(state.amplitude_history) > MAX_CYCLE_HISTORY:
                state.amplitude_history = state.amplitude_history[-MAX_CYCLE_HISTORY:]
                state.suppression_history = state.suppression_history[-MAX_CYCLE_HISTORY:]

    logger.info("Processor thread exiting.")

# ---------------------------------------------------------------------------
# Startup config validation
# ---------------------------------------------------------------------------


def _validate_config(config: Config) -> None:
    """Fail fast if any value required for the live demo is not set in system_config.yaml."""
    missing: list[str] = []
    if config.signal.filter_order is None:
        missing.append("signal.filter_order")
    if config.simulation.timestep_s is None:
        missing.append("simulation.timestep_s")
    if config.simulation.latency_ms is None:
        missing.append("simulation.latency_ms")
    if config.ml.confidence_threshold is None:
        missing.append("ml.confidence_threshold")
    if config.controller.severity_threshold is None:
        missing.append("controller.severity_threshold")
    if config.controller.hysteresis_pct is None:
        missing.append("controller.hysteresis_pct")
    if config.controller.target_suppression_pct is None:
        missing.append("controller.target_suppression_pct")
    if config.controller.suppression_tolerance_pct is None:
        missing.append("controller.suppression_tolerance_pct")
    if config.controller.max_delta_per_step is None:
        missing.append("controller.max_delta_per_step")
    if config.controller.param_bounds is None:
        missing.append("controller.param_bounds")
    if missing:
        raise ValueError(
            "The following config/system_config.yaml values must be set before running "
            "the live demo:\n" + "\n".join(f"  - {k}" for k in missing)
        )

# ---------------------------------------------------------------------------
# Dash application
# ---------------------------------------------------------------------------

app = Dash(__name__, title="Tremor Detection Demo")
app.layout = html.Div(
    [
        # Status banner — shows source type, source label, and pipeline status.
        html.Div(id="status-banner", style={"marginBottom": "10px"}),

        # Row 1: Raw signal | Filtered tremor-band signal
        html.Div(
            [
                dcc.Graph(
                    id="graph-raw",
                    style={"flex": "1"},
                    config={"staticPlot": True},
                ),
                dcc.Graph(
                    id="graph-filtered",
                    style={"flex": "1"},
                    config={"staticPlot": True},
                ),
            ],
            style={"display": "flex", "gap": "10px", "marginBottom": "10px"},
        ),

        # Row 2: PSD | ML + controller status panel
        html.Div(
            [
                dcc.Graph(
                    id="graph-psd",
                    style={"flex": "1"},
                    config={"staticPlot": True},
                ),
                html.Div(
                    id="ml-panel",
                    style={
                        "flex": "1",
                        "backgroundColor": _C["surface"],
                        "borderRadius": "6px",
                        "padding": "24px",
                        "display": "flex",
                        "flexDirection": "column",
                        "justifyContent": "center",
                        "gap": "10px",
                        "border": f"1px solid {_C['border']}",
                    },
                ),
            ],
            style={"display": "flex", "gap": "10px", "marginBottom": "10px"},
        ),

        # Row 3: Stim amplitude over cycles | Suppression over cycles
        html.Div(
            [
                dcc.Graph(
                    id="graph-amplitude",
                    style={"flex": "1"},
                    config={"staticPlot": True},
                ),
                dcc.Graph(
                    id="graph-suppression",
                    style={"flex": "1"},
                    config={"staticPlot": True},
                ),
            ],
            style={"display": "flex", "gap": "10px", "marginBottom": "10px"},
        ),

        # Row 4: Before vs after waveform (full width)
        dcc.Graph(id="graph-before-after", config={"staticPlot": True}),

        # Intervals
        dcc.Interval(id="interval-fast", interval=FAST_INTERVAL_MS, n_intervals=0),
        dcc.Interval(id="interval-slow", interval=SLOW_INTERVAL_MS, n_intervals=0),
    ],
    style={
        "backgroundColor": _C["bg"],
        "color": _C["text"],
        "fontFamily": "monospace, 'Courier New', Courier",
        "padding": "16px",
        "minHeight": "100vh",
    },
)

# ---------------------------------------------------------------------------
# Shared graph layout helpers
# ---------------------------------------------------------------------------


def _base_layout(title: str) -> dict:
    """Return a dict of Plotly figure layout kwargs consistent with the dashboard theme."""
    return {
        "title": {
            "text": title,
            "font": {"color": _C["text"], "size": 13, "family": "monospace"},
            "x": 0,
            "xanchor": "left",
            "pad": {"l": 4},
        },
        "paper_bgcolor": _C["surface"],
        "plot_bgcolor": _C["surface"],
        "font": {"color": _C["text"], "family": "monospace", "size": 11},
        "margin": {"t": 44, "b": 44, "l": 56, "r": 16},
        "xaxis": {
            "gridcolor": _C["border"],
            "zerolinecolor": _C["border"],
            "tickfont": {"color": _C["text_muted"]},
        },
        "yaxis": {
            "gridcolor": _C["border"],
            "zerolinecolor": _C["border"],
            "tickfont": {"color": _C["text_muted"]},
        },
        "legend": {
            "font": {"color": _C["text_muted"], "size": 10},
            "bgcolor": "rgba(0,0,0,0)",
        },
    }


def _waiting_figure(title: str) -> go.Figure:
    """Return a placeholder figure shown before the first data arrives."""
    fig = go.Figure()
    fig.add_annotation(
        text="Waiting for data...",
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font={"color": _C["text_muted"], "size": 13},
    )
    fig.update_layout(**_base_layout(title))
    return fig

# ---------------------------------------------------------------------------
# Callbacks — fast (200 ms): status banner, raw signal, filtered signal
# ---------------------------------------------------------------------------


@app.callback(
    Output("status-banner", "children"),
    Output("graph-raw", "figure"),
    Output("graph-filtered", "figure"),
    Input("interval-fast", "n_intervals"),
)
def _update_fast(n: int):  # type: ignore[return]
    if _state is None:
        banner = html.Div(
            "Initializing...",
            style={"padding": "12px 16px", "borderRadius": "6px",
                   "backgroundColor": _C["surface"], "color": _C["text_muted"]},
        )
        return banner, _waiting_figure("Raw signal"), _waiting_figure("Filtered signal")

    with _state.lock:
        status = _state.status
        countdown = _state.calibration_countdown
        raw_vals = list(_state.raw_deque)
        filt_vals = list(_state.filtered_deque)
        time_vals = list(_state.time_deque)

    # Status banner content and styling.
    source_type = _state.source_type      # immutable — no lock needed
    source_label = _state.source_label

    if status == STATUS_CALIBRATING:
        banner_text = (
            f"SOURCE: {source_type}  --  {source_label}"
            f"   |   STATUS: CALIBRATING  --  hold device still  ({countdown}s remaining)"
        )
        border_color = _C["amber"]
        bg_color = "#1a1800"
    elif status == STATUS_ACTIVE:
        border_color = _C["live_border"] if source_type == SOURCE_LIVE else _C["replay_border"]
        if source_type == SOURCE_LIVE:
            label = "LIVE DATA"
            bg_color = "#0d1f0d"
        else:
            label = "REPLAY -- pre-recorded data"
            bg_color = "#1a1500"
        banner_text = (
            f"SOURCE: {label}  --  {source_label}   |   STATUS: ACTIVE"
        )
    else:  # DISCONNECTED
        banner_text = (
            f"SOURCE: {source_type}  --  {source_label}   |   STATUS: DISCONNECTED"
        )
        border_color = _C["disc_border"]
        bg_color = "#1f0d0d"

    banner = html.Div(
        banner_text,
        style={
            "padding": "12px 16px",
            "borderRadius": "6px",
            "border": f"1px solid {border_color}",
            "backgroundColor": bg_color,
            "color": border_color,
            "fontSize": "13px",
            "letterSpacing": "0.04em",
        },
    )

    # Raw signal figure.
    if raw_vals and time_vals:
        n_raw = len(raw_vals)
        x_raw = list(time_vals)[-n_raw:]
        raw_fig = go.Figure(
            go.Scatter(
                x=x_raw, y=raw_vals,
                mode="lines",
                line={"color": _C["accent"], "width": 1},
                name="accel magnitude",
            )
        )
        raw_fig.update_layout(
            **_base_layout("Raw signal  (accel magnitude, g)"),
            xaxis_title="Time (s)",
            yaxis_title="g",
        )
    else:
        raw_fig = _waiting_figure("Raw signal  (accel magnitude, g)")

    # Filtered signal figure.
    if filt_vals and time_vals:
        n_filt = len(filt_vals)
        n_time = len(time_vals)
        x_filt = list(time_vals)[-min(n_filt, n_time):]
        y_filt = filt_vals[-len(x_filt):]
        filt_fig = go.Figure(
            go.Scatter(
                x=x_filt, y=y_filt,
                mode="lines",
                line={"color": "#ff7b72", "width": 1},
                name="tremor band (4-12 Hz)",
            )
        )
        filt_fig.update_layout(
            **_base_layout("Filtered tremor-band signal  (4-12 Hz, g)"),
            xaxis_title="Time (s)",
            yaxis_title="g",
        )
    else:
        filt_fig = _waiting_figure("Filtered tremor-band signal  (4-12 Hz, g)")

    return banner, raw_fig, filt_fig

# ---------------------------------------------------------------------------
# Callbacks — slow (800 ms): PSD, ML panel, amplitude, suppression, before/after
# ---------------------------------------------------------------------------


@app.callback(
    Output("graph-psd", "figure"),
    Output("ml-panel", "children"),
    Output("graph-amplitude", "figure"),
    Output("graph-suppression", "figure"),
    Output("graph-before-after", "figure"),
    Input("interval-slow", "n_intervals"),
)
def _update_slow(n: int):  # type: ignore[return]
    empty_psd = _waiting_figure("Welch PSD")
    empty_amp = _waiting_figure("Stim amplitude over cycles")
    empty_supp = _waiting_figure("Simulated suppression over cycles (%)")
    empty_ba = _waiting_figure("Real tremor signal vs Simulated suppression response")
    waiting_ml = [html.Div("Waiting for first window...", style={"color": _C["text_muted"]})]

    if _state is None:
        return empty_psd, waiting_ml, empty_amp, empty_supp, empty_ba

    with _state.lock:
        freqs = _state.latest_freqs
        psd = _state.latest_psd
        inference = _state.latest_inference
        stim_params = _state.latest_stim_params
        sim_result = _state.latest_sim_result
        before_win = _state.before_window
        after_win = _state.after_window
        amp_hist = list(_state.amplitude_history)
        supp_hist = list(_state.suppression_history)
        cycle_count = _state.cycle_count
        status = _state.status

    if status == STATUS_CALIBRATING:
        return empty_psd, waiting_ml, empty_amp, empty_supp, empty_ba

    config = _state.config      # immutable — no lock needed
    tremor_band = tuple(config.signal.tremor_band_hz)

    # ---- PSD figure ----
    if freqs is not None and psd is not None:
        psd_fig = go.Figure()
        psd_fig.add_trace(go.Scatter(
            x=freqs, y=psd,
            mode="lines",
            line={"color": _C["accent"], "width": 1.5},
            name="PSD",
        ))
        # Shade the tremor band.
        psd_fig.add_vrect(
            x0=tremor_band[0], x1=tremor_band[1],
            fillcolor=_C["red"],
            opacity=0.12,
            layer="below",
            line_width=0,
            annotation_text="tremor band",
            annotation_font={"color": _C["red"], "size": 10},
        )
        psd_fig.update_layout(
            **_base_layout("Welch PSD  (g^2 / Hz)"),
            xaxis_title="Frequency (Hz)",
            yaxis_title="PSD (g^2/Hz)",
        )
    else:
        psd_fig = empty_psd

    # ---- ML + controller status panel ----
    if inference is None:
        ml_children = waiting_ml
    else:
        label_text = "TREMOR DETECTED" if inference.label else "NO TREMOR"
        label_color = _C["red"] if inference.label else _C["green"]
        controller_active = stim_params is not None

        def _row(label: str, value: str) -> html.Div:
            return html.Div(
                [
                    html.Span(label, style={"color": _C["text_muted"], "minWidth": "160px", "display": "inline-block"}),
                    html.Span(value, style={"color": _C["text"], "fontWeight": "bold"}),
                ],
                style={"fontSize": "13px"},
            )

        dom_freq = (
            f"{inference.dominant_frequency_hz:.1f} Hz"
            if inference.dominant_frequency_hz is not None
            else "N/A"
        )
        supp_text = (
            f"{sim_result.achieved_suppression_pct:.1f}%  (model-predicted)"
            if sim_result is not None
            else "--"
        )
        amp_text = (
            f"{stim_params.amplitude:.3f}  (simulated)"
            if stim_params is not None
            else "--  (no mitigation)"
        )
        ctrl_text = "MITIGATING" if controller_active else "IDLE"
        ctrl_color = _C["amber"] if controller_active else _C["text_muted"]

        ml_children = [
            # Detection label
            html.Div(
                label_text,
                style={
                    "fontSize": "20px",
                    "fontWeight": "bold",
                    "color": label_color,
                    "padding": "10px 18px",
                    "border": f"1px solid {label_color}",
                    "borderRadius": "4px",
                    "marginBottom": "16px",
                    "textAlign": "center",
                    "letterSpacing": "0.05em",
                },
            ),
            _row("Confidence", f"{inference.confidence * 100:.1f}%"),
            _row("Severity", f"{inference.severity:.3f}"),
            _row("Dominant frequency", dom_freq),
            _row("Window amplitude", f"{inference.amplitude:.4f} g"),
            html.Hr(style={"borderColor": _C["border"], "margin": "12px 0"}),
            html.Div(
                f"Controller: {ctrl_text}",
                style={"color": ctrl_color, "fontSize": "13px", "fontWeight": "bold"},
            ),
            _row("Stim amplitude", amp_text),
            _row("Suppression", supp_text),
            html.Div(
                f"Cycle: {cycle_count}",
                style={"color": _C["text_muted"], "fontSize": "11px", "marginTop": "10px"},
            ),
        ]

    # ---- Stim amplitude history ----
    if amp_hist:
        x_cycles = list(range(max(0, cycle_count - len(amp_hist)), cycle_count))
        amp_fig = go.Figure(
            go.Scatter(
                x=x_cycles, y=amp_hist,
                mode="lines+markers",
                line={"color": _C["amber"], "width": 1.5},
                marker={"size": 4, "color": _C["amber"]},
                name="amplitude",
            )
        )
        amp_fig.update_layout(
            **_base_layout("Stim amplitude over cycles  (simulated)"),
            xaxis_title="Cycle",
            yaxis_title="Amplitude (a.u.)",
            yaxis_range=[0, config.controller.param_bounds.amplitude.max * 1.05],
        )
    else:
        amp_fig = empty_amp

    # ---- Suppression history ----
    if supp_hist:
        x_cycles = list(range(max(0, cycle_count - len(supp_hist)), cycle_count))
        target_pct: float = config.controller.target_suppression_pct
        supp_fig = go.Figure()
        supp_fig.add_trace(go.Scatter(
            x=x_cycles, y=supp_hist,
            mode="lines+markers",
            line={"color": _C["green"], "width": 1.5},
            marker={"size": 4, "color": _C["green"]},
            name="suppression",
        ))
        # Target line.
        supp_fig.add_hline(
            y=target_pct,
            line_dash="dot",
            line_color=_C["text_muted"],
            annotation_text=f"target {target_pct:.0f}%",
            annotation_font={"color": _C["text_muted"], "size": 10},
        )
        supp_fig.update_layout(
            **_base_layout("Simulated suppression over cycles  (model-predicted, %)"),
            xaxis_title="Cycle",
            yaxis_title="Suppression (%)",
            yaxis_range=[0, 105],
        )
    else:
        supp_fig = empty_supp

    # ---- Before vs after waveform ----
    if before_win is not None and after_win is not None:
        n_samples = len(before_win)
        t_win = np.arange(n_samples, dtype=np.float64) / config.sensor.sample_rate_hz
        ba_fig = go.Figure()
        ba_fig.add_trace(go.Scatter(
            x=t_win, y=before_win,
            mode="lines",
            line={"color": _C["red"], "width": 1.5},
            name="Real tremor signal",
        ))
        ba_fig.add_trace(go.Scatter(
            x=t_win, y=after_win,
            mode="lines",
            line={"color": _C["green"], "width": 1.5},
            name="Simulated suppression response",
        ))
        note = (
            f"Simulated suppression: {sim_result.achieved_suppression_pct:.1f}%  --  "
            "model-predicted reduction, not a physical measurement"
            if sim_result is not None
            else "Controller idle  --  no suppression applied this cycle"
        )
        ba_fig.add_annotation(
            text=note,
            x=0.5, y=1.06,
            xref="paper", yref="paper",
            showarrow=False,
            font={"color": _C["text_muted"], "size": 11},
        )
        ba_fig.update_layout(
            **_base_layout(
                "Real tremor signal  vs  Simulated suppression response  (model-predicted)"
            ),
            xaxis_title="Window time (s)",
            yaxis_title="g",
        )
    else:
        ba_fig = empty_ba

    return psd_fig, ml_children, amp_fig, supp_fig, ba_fig

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Live tremor detection and adaptive control demonstration dashboard. "
            "Opens at http://127.0.0.1:8050"
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--port",
        metavar="PORT",
        help="Serial port for live ESP32 data (e.g. COM5 on Windows, /dev/ttyUSB0 on Linux).",
    )
    source_group.add_argument(
        "--replay",
        metavar="CSV_PATH",
        help="Path to a pre-recorded raw_stream.csv to replay at real-time pacing.",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        metavar="PATH",
        help=f"Path to trained model artifact (.pkl). Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--scaler-path",
        default=str(DEFAULT_SCALER_PATH),
        metavar="PATH",
        help=f"Path to fitted StandardScaler (.pkl). Default: {DEFAULT_SCALER_PATH}",
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    config = load_config()
    _validate_config(config)

    model_path = Path(args.model_path)
    scaler_path = Path(args.scaler_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}\n"
            "Run scripts/run_training.py first to train and save the model."
        )
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler artifact not found: {scaler_path}\n"
            "Run scripts/run_training.py first to train and save the scaler."
        )

    # Build the data source and shared state.
    if args.port:
        source = SerialSource(args.port)
        source_type = SOURCE_LIVE
        source_label = args.port
    else:
        replay_path = Path(args.replay)
        source = ReplaySource(replay_path, rate_hz=config.sensor.sample_rate_hz)
        source_type = SOURCE_REPLAY
        source_label = str(replay_path)

    global _state
    _state = SharedState(source_type=source_type, source_label=source_label, config=config)

    sample_queue: queue.Queue = queue.Queue(maxsize=2000)

    # Start daemon threads — they exit automatically when the main process exits.
    reader = threading.Thread(
        target=_reader_thread_fn,
        args=(source, sample_queue, _state),
        daemon=True,
        name="demo-reader",
    )
    processor = threading.Thread(
        target=_processor_thread_fn,
        args=(sample_queue, _state, config, model_path, scaler_path),
        daemon=True,
        name="demo-processor",
    )

    reader.start()
    logger.info("Reader thread started.")
    processor.start()
    logger.info("Processor thread started.")

    logger.info(
        "Dashboard starting — open http://127.0.0.1:8050 in your browser."
    )
    logger.info("Source: %s -- %s", source_type, source_label)

    # debug=False, use_reloader=False: prevents Werkzeug from spawning a second
    # process, which would start a second reader/processor thread (see library-docs.md).
    app.run(debug=False, use_reloader=False, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main()
