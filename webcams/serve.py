"""Run the local two-webcam page."""
import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
from threading import Lock

from experiment import Experiment

WEB_ROOT = Path(__file__).resolve().parent
EXPOSURE_SCRIPTS = {
    "darwin": WEB_ROOT.parent / "camera-exposure-macos.sh",
    "linux": WEB_ROOT.parent / "camera-exposure.sh",
}
exposure_lock = Lock()
experiment = Experiment()


def apply_exposure():
    script = EXPOSURE_SCRIPTS.get(sys.platform)
    if script is None:
        return False, "No camera exposure script for this operating system."
    with exposure_lock:
        try:
            result = subprocess.run(
                ["bash", str(script)], capture_output=True, text=True, timeout=25
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return False, str(error)
    details = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    return result.returncode == 0, details


class WebcamHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self):
        if self.path == "/experiment/status":
            self.reply_json(experiment.available())
            return
        super().do_GET()

    def do_POST(self):
        if self.path not in {"/camera-exposure", "/experiment/chat"}:
            self.send_error(404)
            return
        port = self.server.server_port
        origins = {f"http://localhost:{port}", f"http://127.0.0.1:{port}"}
        if self.headers.get("Origin") not in origins:
            self.send_error(403)
            return
        if self.path == "/experiment/chat":
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 12 * 1024 * 1024:
                    raise ValueError("Experiment request is missing or too large.")
                request = json.loads(self.rfile.read(size))
                if not isinstance(request, dict):
                    raise ValueError("Expected a JSON object.")
                self.reply_json(experiment.chat(request))
            except (ValueError, TypeError) as error:
                self.reply_json({"error": str(error)}, 400)
            return
        ok, details = apply_exposure()
        if not ok:
            self.log_error("Camera exposure: %s", details)
        self.reply_json({"ok": ok}, 200 if ok else 503)

    def reply_json(self, value, status=200):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), WebcamHandler) as server:
        ok, details = apply_exposure()
        print(f"Camera exposure {'applied' if ok else 'failed'}: {details}", flush=True)
        print(f"Webcams: http://localhost:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
