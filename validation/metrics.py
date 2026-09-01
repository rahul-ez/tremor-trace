"""Validation metrics computed from per-cycle experiment records.

Feature 47 of the build plan. Operates on lists of dicts shaped like
validation/experiments/common.py::EXPERIMENT_RECORD_COLUMNS, as produced by
Features 44-46's experiment runs (and read back from a written
cycle_log.csv).

tremor-power reduction and residual amplitude are not separate functions
here -- they are the achieved_suppression_pct/residual_amplitude fields
already computed per-cycle by simulation/suppression.py (Feature 34) and
simulation/stimulation_model.py (Feature 33); this module aggregates them
across a run rather than recomputing them.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _suppression_values(records: list[dict]) -> list[float]:
    if not records:
        raise ValueError("records must be non-empty")
    return [float(r["achieved_suppression_pct"]) for r in records]


def mean_suppression_pct(records: list[dict]) -> float:
    """Mean tremor-power-reduction (suppression) percentage across all cycles.

    Non-mitigating cycles contribute 0.0 (their recorded
    achieved_suppression_pct, per validation/experiments/common.py), so
    this reflects overall run performance, not just performance while
    actively mitigating.

    Args:
        records: Non-empty list of per-cycle records.

    Returns:
        Mean achieved_suppression_pct across all records.
    """
    values = _suppression_values(records)
    return sum(values) / len(values)


def mean_residual_amplitude(records: list[dict]) -> float:
    """Mean residual tremor amplitude (units g) across all cycles.

    Args:
        records: Non-empty list of per-cycle records.

    Returns:
        Mean residual_amplitude across all records.
    """
    if not records:
        raise ValueError("records must be non-empty")
    return sum(float(r["residual_amplitude"]) for r in records) / len(records)


def time_to_target(
    records: list[dict],
    target_suppression_pct: float,
    tolerance_pct: float,
    cycle_duration_s: float,
) -> Optional[float]:
    """Time (seconds) until achieved_suppression_pct first enters the target band.

    Args:
        records: Per-cycle records, in cycle order.
        target_suppression_pct: Target suppression percentage.
        tolerance_pct: Half-width of the acceptance band around the target
            (matches config.controller.suppression_tolerance_pct).
        cycle_duration_s: Duration of one cycle in seconds
            (config.signal.window_length_s), used to convert the cycle
            index into elapsed time.

    Returns:
        Elapsed time in seconds at the first cycle whose
        achieved_suppression_pct falls within
        [target - tolerance, target + tolerance], or None if the run never
        entered that band.
    """
    low, high = target_suppression_pct - tolerance_pct, target_suppression_pct + tolerance_pct
    for record in records:
        if low <= float(record["achieved_suppression_pct"]) <= high:
            return record["cycle"] * cycle_duration_s
    return None


def mitigation_duty_cycle_pct(records: list[dict]) -> float:
    """Fraction of run time spent actively mitigating, as a percentage.

    Distinct from StimParams.duty_cycle (the electrical waveform's own
    on/off duty cycle when active) -- this measures how much of the whole
    experiment's wall-clock time the controller chose to mitigate at all.

    Args:
        records: Non-empty list of per-cycle records.

    Returns:
        100 * (mitigating cycles / total cycles).
    """
    if not records:
        raise ValueError("records must be non-empty")
    mitigating = sum(1 for r in records if r["mitigate"])
    return 100.0 * mitigating / len(records)


def total_simulated_exposure(records: list[dict], cycle_duration_s: float) -> float:
    """Integrated stimulation "dose": sum of amplitude over mitigating time.

    Amplitude only (not amplitude * duty_cycle, despite duty_cycle being a
    documented StimParams field): confirmed against this project's actual
    controller behavior (controller/parameter_selection.py::select_initial_params
    pins duty_cycle to config.controller.param_bounds.duty_cycle.min, and
    adapt_params() never adjusts it -- "amplitude is the only adapted
    field" by design), so under the current v1 config duty_cycle is
    always 0.0 and multiplying by it would make this metric trivially
    zero for every adaptive run regardless of actual amplitude used,
    silently defeating the Feature 46 exposure comparison it exists to
    support. If duty_cycle ever becomes a real adapted lever, revisit this
    formula.

    Args:
        records: Per-cycle records.
        cycle_duration_s: Duration of one cycle in seconds.

    Returns:
        Total simulated exposure (arbitrary units: amplitude-seconds).
        Non-mitigating cycles (amplitude is None) contribute 0.
    """
    total = 0.0
    for record in records:
        amplitude = record.get("amplitude")
        if amplitude is None:
            continue
        total += float(amplitude) * cycle_duration_s
    return total


def false_activation_rate(records: list[dict], ground_truth_labels: list[bool]) -> float:
    """Fraction of cycles where mitigation activated despite no real tremor.

    Args:
        records: Per-cycle records, in cycle order.
        ground_truth_labels: One bool per record, aligned by position (True
            = a real tremor was genuinely present in that cycle's window).
            Required -- this metric cannot be computed without ground
            truth, which most synthetic/unlabeled runs do not have.

    Returns:
        100 * (cycles where mitigate=True and ground_truth=False) / (total
        cycles where ground_truth=False). Returns 0.0 if there are no
        ground_truth=False cycles at all (nothing to falsely activate on).

    Raises:
        ValueError: If records and ground_truth_labels have different lengths.
    """
    if len(records) != len(ground_truth_labels):
        raise ValueError(
            f"records has {len(records)} entries but ground_truth_labels has "
            f"{len(ground_truth_labels)}; they must be aligned 1:1"
        )
    no_tremor_cycles = [r for r, truth in zip(records, ground_truth_labels) if not truth]
    if not no_tremor_cycles:
        return 0.0
    false_activations = sum(1 for r in no_tremor_cycles if r["mitigate"])
    return 100.0 * false_activations / len(no_tremor_cycles)


def mean_detection_to_actuation_latency_ms(records: list[dict]) -> float:
    """Mean reported detection-to-actuation latency across mitigating cycles.

    Currently a direct pass-through of config.simulation.latency_ms
    (constant per run, not yet cycle-varying), reported here as a proper
    aggregate metric rather than a raw config value, so it stands
    consistently alongside the other metrics in a validation report.

    Args:
        records: Per-cycle records.

    Returns:
        Mean latency_ms across cycles where mitigation occurred, or 0.0 if
        no cycle mitigated.
    """
    mitigating = [r for r in records if r["mitigate"]]
    if not mitigating:
        return 0.0
    return sum(float(r["latency_ms"]) for r in mitigating) / len(mitigating)


def controller_stability_oscillation(records: list[dict]) -> float:
    """Mean absolute cycle-to-cycle change in stimulation amplitude.

    A simple, defensible measure of controller stability: a well-converged
    adaptive controller should settle into small or zero changes between
    consecutive cycles once near target; large sustained values indicate
    oscillation/instability (architecture.md -> Adaptive Controller
    explicitly requires no unbounded oscillation).

    Args:
        records: Per-cycle records, in cycle order.

    Returns:
        Mean |amplitude[i] - amplitude[i-1]| over consecutive cycle pairs
        that both have a non-None amplitude. 0.0 if fewer than two such
        pairs exist.
    """
    amplitudes = [r["amplitude"] for r in records if r["amplitude"] is not None]
    if len(amplitudes) < 2:
        return 0.0
    deltas = [abs(amplitudes[i] - amplitudes[i - 1]) for i in range(1, len(amplitudes))]
    return sum(deltas) / len(deltas)
