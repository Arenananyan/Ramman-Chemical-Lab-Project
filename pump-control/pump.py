"""Control the Arduino Nano pump using board-timed pulses."""

import argparse
from contextlib import suppress
import math
import sys
import time

import serial


class PumpError(Exception):
    """The Arduino did not confirm the expected command."""


def send(port, command):
    if port.write(command) != len(command):
        raise PumpError("Incomplete serial write")


def expect(port, expected):
    response = port.read_until(size=64)
    if not response.endswith(b"\n") or response.strip() != expected:
        raise PumpError(f"Expected {expected!r}, received {response!r}")


def stop(port):
    send(port, b"S")
    expect(port, b"Pump OFF")


def pulse(port, seconds):
    if not math.isfinite(seconds) or not 0.1 <= seconds <= 2.0:
        raise ValueError("Pulse duration must be between 0.1 and 2.0 seconds")

    duration_ms = round(seconds * 1000)
    send(port, b"?")
    expect(port, b"PUMP V1")
    try:
        send(port, f"P{duration_ms:04d}\n".encode("ascii"))
        expect(port, b"Pump ON")
        port.timeout = seconds + 1.0
        expect(port, b"Pump DONE")
    except (OSError, PumpError, KeyboardInterrupt):
        # The Nano times the pulse even when the connection is unavailable.
        with suppress(OSError, PumpError):
            send(port, b"S")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="Arduino serial device")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("off", help="Switch the pump off and check the reply")
    pulse_parser = commands.add_parser("pulse", help="Run a supervised timed pulse")
    pulse_parser.add_argument("seconds", type=float)
    args = parser.parse_args()

    try:
        with serial.Serial(
            args.port, 9600, timeout=0.5, write_timeout=0.5, exclusive=True
        ) as port:
            # Opening the port can reset a Nano. Wait for the sketch to start.
            time.sleep(2.0)
            port.reset_input_buffer()
            stop(port)
            if args.command == "pulse":
                pulse(port, args.seconds)
                print("Timed pulse completed (Nano reports pump off)")
            else:
                print("Pump OFF acknowledged")
    except (serial.SerialException, OSError, PumpError, ValueError) as error:
        print(f"Pump command failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Pump command interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
