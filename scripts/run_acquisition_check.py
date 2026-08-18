"""PC-Side Raw Data Logger script for ESP32 serial acquisition stream.

Reads incoming USB serial CSV lines (or stdin/file stream) from the ESP32 MPU6050 acquisition system,
validates sample formatting, discards error marker lines (ERR,*), and logs clean raw data to:
data/raw/<subject_id>/<session_id>/raw_stream.csv
"""

import argparse
import logging
from pathlib import Path
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_HEADER = "timestamp_us,ax,ay,az,gx,gy,gz"


def find_available_port() -> str | None:
    """Auto-detect available serial ports."""
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            logger.warning("No serial ports detected on the system.")
            return None
        if len(ports) == 1:
            detected_port = ports[0].device
            logger.info("Auto-detected single serial port: %s (%s)", detected_port, ports[0].description)
            return detected_port

        logger.info("Multiple serial ports found:")
        for p in ports:
            logger.info("  - %s (%s)", p.device, p.description)
        return None
    except ImportError:
        return None


def parse_csv_line(line: str) -> tuple[bool, str]:
    """Validate a CSV sample line.

    Returns:
        (is_valid, cleaned_line)
    """
    line = line.strip()
    if not line:
        return False, ""
    if line.startswith("ERR,"):
        logger.warning("Received firmware error marker: %s", line)
        return False, ""
    if line == EXPECTED_HEADER:
        return False, ""  # Skip header line if present in stream

    parts = line.split(",")
    if len(parts) != 7:
        logger.warning("Discarding malformed line (expected 7 fields): %s", line)
        return False, ""

    try:
        timestamp_us = int(parts[0])
        ax = int(parts[1])
        ay = int(parts[2])
        az = int(parts[3])
        gx = int(parts[4])
        gy = int(parts[5])
        gz = int(parts[6])
    except ValueError:
        logger.warning("Discarding malformed line (parse error): %s", line)
        return False, ""

    cleaned = f"{timestamp_us},{ax},{ay},{az},{gx},{gy},{gz}"
    return True, cleaned


def run_logger(
    output_path: Path,
    port: str | None = None,
    baud: int = 115200,
    duration_s: float | None = None,
    input_stream=None,
) -> int:
    """Run data logger reading from serial port or stream, saving to output_path.

    Returns:
        Number of valid samples logged.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stream = input_stream
    serial_obj = None

    if stream is None:
        selected_port = port if port is not None else find_available_port()

        if selected_port is not None:
            try:
                import serial
                serial_obj = serial.Serial(selected_port, baud, timeout=1.0)
                stream = serial_obj
                logger.info("Successfully opened serial port %s at %d baud", selected_port, baud)
            except Exception as e:
                logger.error(
                    "Failed to open serial port %s: %s.\n"
                    "Note: If another application (Serial Studio, Arduino Monitor, VSCode Terminal) is currently using %s, "
                    "please close it and run this script again.",
                    selected_port,
                    e,
                    selected_port,
                )
                raise
        else:
            logger.info("No serial port specified or auto-selected. Reading from standard input (stdin)...")
            stream = sys.stdin

    logger.info("Logging data to %s", output_path)

    valid_count = 0
    start_time = time.time()
    last_status_time = start_time

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(EXPECTED_HEADER + "\n")
        f.flush()

        try:
            while True:
                now = time.time()
                if duration_s is not None and (now - start_time) >= duration_s:
                    logger.info("Duration limit of %.1fs reached", duration_s)
                    break

                if serial_obj is not None:
                    raw_line = serial_obj.readline().decode("utf-8", errors="replace")
                elif hasattr(stream, "readline"):
                    raw_line = stream.readline()
                    if not raw_line:  # EOF
                        break
                    if isinstance(raw_line, bytes):
                        raw_line = raw_line.decode("utf-8", errors="replace")
                else:
                    break

                is_valid, cleaned_line = parse_csv_line(raw_line)
                if is_valid:
                    f.write(cleaned_line + "\n")
                    f.flush()
                    valid_count += 1

                    if now - last_status_time >= 5.0:
                        logger.info("Logged %d valid samples so far...", valid_count)
                        last_status_time = now

        except KeyboardInterrupt:
            logger.info("Acquisition stopped by user")

    if serial_obj is not None:
        serial_obj.close()

    logger.info("Logging complete. Wrote %d valid samples to %s", valid_count, output_path)
    return valid_count


def main() -> None:
    parser = argparse.ArgumentParser(description="PC-side raw IMU data logger")
    parser.add_argument("--subject-id", type=str, default="subj01", help="Subject ID")
    parser.add_argument("--session-id", type=str, default="sess01", help="Session ID")
    parser.add_argument("--port", type=str, default=None, help="Serial port (e.g. COM9 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--duration", type=float, default=None, help="Duration to record in seconds")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "data" / "raw" / args.subject_id / args.session_id / "raw_stream.csv"

    run_logger(output_path, port=args.port, baud=args.baud, duration_s=args.duration)


if __name__ == "__main__":
    main()
