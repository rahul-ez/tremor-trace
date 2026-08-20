"""Integration test for the Phase 3 baseline visualization script."""

from pathlib import Path

import pytest

from scripts.run_pipeline_baseline import run_pipeline


def test_run_pipeline_saves_baseline_plot(tmp_path: Path) -> None:
    input_path = Path("data/raw/subj01/sess01/raw_stream.csv")
    if not input_path.exists():
        pytest.skip(f"Recorded session not found at {input_path}")

    output_path = tmp_path / "pipeline_baseline.png"

    result_path = run_pipeline(
        input_path=input_path,
        output_path=output_path,
        subject_id="subj01",
        session_id="sess01",
        window_index=0,
    )

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0