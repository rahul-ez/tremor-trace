"""Tests for PC-side raw data logger script."""

import io
from pathlib import Path

from scripts.run_acquisition_check import parse_csv_line, run_logger


def test_parse_csv_line_valid() -> None:
    is_valid, cleaned = parse_csv_line("1000000,100,200,16000,10,20,30\n")
    assert is_valid is True
    assert cleaned == "1000000,100,200,16000,10,20,30"


def test_parse_csv_line_err_marker() -> None:
    is_valid, cleaned = parse_csv_line("ERR,I2C_READ_FAILED\n")
    assert is_valid is False
    assert cleaned == ""


def test_parse_csv_line_header() -> None:
    is_valid, cleaned = parse_csv_line("timestamp_us,ax,ay,az,gx,gy,gz\n")
    assert is_valid is False
    assert cleaned == ""


def test_parse_csv_line_malformed() -> None:
    is_valid, cleaned = parse_csv_line("1000,100,200\n")
    assert is_valid is False
    assert cleaned == ""

    is_valid, cleaned = parse_csv_line("1000,100,200,abc,10,20,30\n")
    assert is_valid is False
    assert cleaned == ""


def test_run_logger_mock_stream(tmp_path: Path) -> None:
    mock_input = io.StringIO(
        "timestamp_us,ax,ay,az,gx,gy,gz\n"
        "1000000,100,200,16000,10,20,30\n"
        "ERR,I2C_READ_FAILED\n"
        "1010000,102,201,16005,11,21,31\n"
        "malformed_line\n"
        "1020000,105,203,16010,12,22,32\n"
    )
    output_file = tmp_path / "raw_stream.csv"

    count = run_logger(output_file, input_stream=mock_input)
    assert count == 3
    assert output_file.exists()

    content = output_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 4  # Header + 3 valid rows
    assert content[0] == "timestamp_us,ax,ay,az,gx,gy,gz"
    assert content[1] == "1000000,100,200,16000,10,20,30"
    assert content[2] == "1010000,102,201,16005,11,21,31"
    assert content[3] == "1020000,105,203,16010,12,22,32"


def test_run_logger_appends_without_rewriting_existing_data(tmp_path: Path) -> None:
    output_file = tmp_path / "raw_stream.csv"
    output_file.write_text(
        "timestamp_us,ax,ay,az,gx,gy,gz\n"
        "1000000,100,200,16000,10,20,30\n",
        encoding="utf-8",
    )

    count = run_logger(
        output_file,
        input_stream=io.StringIO("1010000,102,201,16005,11,21,31\n"),
    )

    assert count == 1
    content = output_file.read_text(encoding="utf-8").strip().splitlines()
    assert content == [
        "timestamp_us,ax,ay,az,gx,gy,gz",
        "1000000,100,200,16000,10,20,30",
        "1010000,102,201,16005,11,21,31",
    ]
