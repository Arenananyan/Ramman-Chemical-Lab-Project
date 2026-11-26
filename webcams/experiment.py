"""Local experiment coordinator and its three equipment adapters."""
import base64
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
from threading import Lock
import time
from uuid import uuid4

from groq_coordinator import CoordinatorError, GroqCoordinator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "printer-control"))
from printer_client import MockPrinter, PrinterError, SSHPrinter, checked, run_prepared
from printer_policy import same_position

PROFILE = ROOT / "profiles/experiment.local.json"
CALIBRATION = ROOT / "vision/raman-a4tech-20261124.json"
DEMO_TARGET = [135.0, 125.0, 40.0]
PULSE_SECONDS = 0.5
START = {"start", "start demo", "run demo", "start experiment"}
APPROVE = {"yes", "yes position is correct", "yes its right", "yes the position is right",
           "approve", "position is correct"}
REJECT = {"no", "no its wrong", "reject", "position is wrong", "cancel"}


def vector(values, count):
    if not isinstance(values, list) or len(values) != count or any(
        type(value) not in (int, float) or not math.isfinite(value) for value in values
    ):
        raise ValueError(f"Expected {count} finite numbers in the experiment profile or fit.")
    return values


def load_profile():
    if not PROFILE.is_file():
        raise ValueError("Live mode requires a measured profiles/experiment.local.json.")
    data = json.loads(PROFILE.read_text())
    if not isinstance(data, dict) or data.get("verified_for_live") is not True:
        raise ValueError("Verify the camera registration, nozzle offset and clear travel before enabling live mode.")
    matrix = data["printer_xy_from_field"]
    if not isinstance(matrix, list) or len(matrix) != 2:
        raise ValueError("Provide two XY transform rows.")
    a, b, _ = vector(matrix[0], 3)
    c, d, _ = vector(matrix[1], 3)
    if any(abs(error) > 0.02 for error in (a*a+b*b-1, c*c+d*d-1, a*c+b*d)):
        raise ValueError("The XY transform must use measured millimetres with orthogonal unit axes.")
    xmin, xmax, ymin, ymax = vector(data["field_bounds_mm"], 4)
    if not xmin < xmax or not ymin < ymax:
        raise ValueError("Field bounds must be in increasing order.")
    paper, clearance = vector([data["printer_z_at_paper_mm"], data["clearance_z_mm"]], 2)
    if clearance <= paper + 10:
        raise ValueError("Clearance Z must be above the field and the test object.")
    if not os.environ.get("SSH_HOST") or os.environ.get("PRINTER_ALLOW_MOTION") != "1":
        raise ValueError("Start the local server with SSH_HOST and PRINTER_ALLOW_MOTION=1 for live mode.")
    return data


def command(argv, timeout):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        details = result.stderr.strip() or result.stdout.strip()
        raise ValueError(details.splitlines()[-1][:450] if details else "Device command failed.")
    return result.stdout


def locate(frames, profile, folder):
    if not isinstance(frames, list) or len(frames) != 2 or not all(isinstance(frame, str) for frame in frames):
        raise ValueError("Connect both cameras and capture a calibrated pair.")
    size = tuple(json.loads(CALIBRATION.read_text())["image_size"])
    folder.mkdir(parents=True)
    paths = [folder / f"{side}.png" for side in ("left", "right")]
    for path, frame in zip(paths, frames):
        image = base64.b64decode(frame, validate=True)
        if (len(image) < 24 or image[:8] != b"\x89PNG\r\n\x1a\n" or image[12:16] != b"IHDR"
                or struct.unpack(">II", image[16:24]) != size):
            raise ValueError(f"Both frames must match the calibration at {size[0]} × {size[1]}.")
        path.write_bytes(image)

    command([sys.executable, str(ROOT / "vision/cylinder_fit.py"), *map(str, paths),
             "--calibration", str(CALIBRATION), "--field-bounds",
             *map(str, profile["field_bounds_mm"]), "--output-dir", str(folder)], timeout=120)
    x, y, height = vector(json.loads((folder / "measurements.json").read_text())["top_center_mm"], 3)
    gap = 10.0
    target = [a*x + b*y + offset for a, b, offset in profile["printer_xy_from_field"]]
    target.append(profile["printer_z_at_paper_mm"] + height + gap)
    return [x, y, height], target


def position(printer):
    state = checked(printer, {"action": "status"})["state"]
    if (state["webhooks"]["state"] != "ready" or set(state["toolhead"]["homed_axes"]) != set("xyz")
            or state["print_stats"]["state"] not in {"standby", "complete"}
            or abs(state["motion_report"]["live_velocity"]) > 0.001):
        raise ValueError("Printer must be ready, idle and homed on XYZ.")
    return state["motion_report"]["live_position"][:3]


def move_to_object(printer, target, clearance):
    start = position(printer)
    target = [round(value, 3) for value in target]
    if clearance <= target[2]:
        raise ValueError("The fitted top plus the 10 mm tip gap exceeds the verified clearance.")
    transit = max(round(clearance, 3), round(start[2], 3))
    dx, dy = [round(target[i] - start[i], 3) for i in (0, 1)]
    steps = [
        {"axis": "z", "distance_mm": round(transit - start[2], 3)},
        {"axis": "xy", "distance_mm": [dx, dy]},
        {"axis": "z", "distance_mm": round(target[2] - transit, 3)},
    ]
    steps = [step for step in steps if step["distance_mm"] not in (0, [0, 0])]
    speed = 5
    prepared = checked(printer, {"action": "preview", "steps": steps,
                                "speed_mm_s": speed, "z_speed_mm_s": speed})["plan"]
    if not same_position(prepared["start"], start) or not same_position(prepared["end"], target):
        raise ValueError("The printer position changed while planning. Start a new run.")
    result = run_prepared(printer, steps, speed, prepared)
    if not same_position(result["reported_position"], target) or not same_position(position(printer), target):
        raise ValueError("The printer did not confirm the target position. Inspect it before restarting.")
    return target


def coordinates(values):
    return ", ".join(f"{axis} {value:.1f}" for axis, value in zip("XYZ", values)) + " mm"


class Experiment:
    def __init__(self):
        self.lock = Lock()
        self.run = None
        self.coordinator = GroqCoordinator()

    def available(self):
        llm = {"groq_model": self.coordinator.model if self.coordinator.configured else None}
        try:
            load_profile()
        except (OSError, ValueError, KeyError) as error:
            return {"live_available": False, "reason": str(error), **llm}
        return {"live_available": True, **llm}

    def chat(self, request):
        text = request.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > 120:
            raise ValueError("Enter a message of up to 120 characters.")
        intent = re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()
        if not self.lock.acquire(blocking=False):
            raise ValueError("An equipment command is in progress.")
        try:
            return self.respond(intent, request)
        finally:
            self.lock.release()

    def respond(self, intent, request):
        run = self.run
        if run and run["phase"] == "awaiting" and time.monotonic() > run["expires"]:
            run["phase"] = "expired"
        if intent in START:
            if run and run["phase"] == "awaiting":
                raise ValueError("Resolve the current position check before starting another run.")
            return self.start(request)
        if intent in APPROVE | REJECT:
            if run and run["phase"] == "expired" and request.get("run_id") == run["id"]:
                return self.reply(run, ("Coordinator", "The position check expired. Start a new run."))
            if not run or run["phase"] != "awaiting" or request.get("run_id") != run["id"]:
                raise ValueError("There is no matching position check to approve. Start a new run.")
            run["phase"] = "cancelled"
            if intent in REJECT:
                return self.reply(run, ("Coordinator", "Run cancelled. The pump was not activated."))
            return self.dispense(run)
        if intent == "status":
            return self.reply(run, ("Coordinator", f"Run status: {run['phase']}." if run else "Ready for a demo."))
        if self.coordinator.configured:
            context = {"phase": run["phase"] if run else "idle",
                       "mode": run["mode"] if run else request.get("mode"),
                       "observation": run.get("observation") if run else None}
            try:
                action, answer = self.coordinator.chat(request["text"], context)
            except CoordinatorError as error:
                return self.reply(run, ("Coordinator", str(error)))
            if action == "suggest_start":
                answer += " Click Start demo to begin."
            if run and run["phase"] == "awaiting":
                answer += " Use the position buttons or answer yes or no to decide."
            return self.reply(run, ("Groq coordinator", answer))
        return self.reply(run, ("Coordinator", "Say “start demo” or “status”. Once the head arrives, say “yes” or “no” to confirm its position."))

    def start(self, request):
        mode = request.get("mode")
        if mode not in {"simulation", "live"}:
            raise ValueError("Choose simulation or live mode.")
        run = {"id": uuid4().hex, "phase": "positioning", "mode": mode}
        self.run = run
        messages = []
        try:
            if mode == "simulation":
                printer = MockPrinter()
                field, target, clearance = [35, 25, 30], DEMO_TARGET, 60
                messages.append(("Vision", f"Simulated cylinder top at {coordinates(field)}."))
            else:
                if request.get("verified_setup") is not True:
                    raise ValueError("Check the live setup before starting.")
                profile = load_profile()
                printer = SSHPrinter()
                folder = ROOT / "runs" / "experiment" / run["id"]
                field, target = locate(request.get("frames"), profile, folder)
                clearance = profile["clearance_z_mm"]
                messages.append(("Vision", f"Fitted cylinder top at {coordinates(field)} in the camera field."))
            observation = {
                "mode": mode, "cylinder_top_in_field_mm": field,
                "registered_target_in_printer_mm": target, "clearance_z_mm": clearance,
            }
            run["observation"] = observation
            if self.coordinator.configured:
                action, summary = self.coordinator.handoff(observation)
                messages.append(("Groq coordinator", summary))
                if action == "hold":
                    raise ValueError("The Groq coordinator held the vision-to-motion handoff.")
            target = move_to_object(printer, target, clearance)
        except (OSError, PrinterError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
            run["phase"] = "failed"
            return self.reply(run, *messages, ("Coordinator", f"Run stopped before pumping. {error}"))
        run.update(phase="awaiting", printer=printer, target=target, expires=time.monotonic() + 120)
        return self.reply(run, *messages, ("Motion", f"Head reported at {coordinates(target)}."),
                          ("Coordinator", "Is the head in the right position over the object? Check both views, then answer yes or no."))

    def dispense(self, run):
        run["phase"] = "failed"
        try:
            if not same_position(position(run["printer"]), run["target"]):
                raise ValueError("The head moved after the position check.")
            if run["mode"] == "live":
                command([sys.executable, str(ROOT / "pump-control/pump.py"),
                         "--port", os.environ.get("PUMP_PORT", "/dev/cu.usbserial-110"),
                         "pulse", str(PULSE_SECONDS)], timeout=12)
        except (OSError, PrinterError, ValueError, subprocess.TimeoutExpired) as error:
            return self.reply(run, ("Pump", f"Pulse could not be confirmed. Inspect the setup before trying again. {error}"))
        run["phase"] = "completed"
        result = "Simulated a 0.5-second pulse." if run["mode"] == "simulation" else "Nano completed the 0.5-second pulse and reported pump off."
        return self.reply(run, ("Pump", result), ("Coordinator", "Run complete."))

    @staticmethod
    def reply(run, *messages):
        return {"run_id": run["id"] if run else None, "phase": run["phase"] if run else "idle",
                "messages": [{"role": role, "text": text} for role, text in messages]}
