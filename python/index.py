"""
Internal Vercel Python service entrypoint.
Exposes /health and /sources/<filename> for the bound Express API service
using Python's standard library (zero heavy serverless wheel bloat).
"""

from http.server import BaseHTTPRequestHandler
import json
import os
from urllib.parse import unquote, urlparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ALLOWED_SOURCE_FILES = {
    "object_detector.py",
    "setup_models.py",
    "requirements-desktop.txt",
    "requirements.txt",
    "README.md",
}


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"

        if route in ("/", "/health"):
            self._send_json(200, {"status": "ok", "service": "python"})
            return

        if route.startswith("/sources/"):
            filename = unquote(route[len("/sources/") :])
            if filename not in ALLOWED_SOURCE_FILES:
                self._send_json(404, {"error": "File not found."})
                return

            target_path = os.path.join(BASE_DIR, filename)
            if not os.path.exists(target_path):
                self._send_json(404, {"error": "File does not exist."})
                return

            with open(target_path, "r", encoding="utf-8") as f:
                content = f.read()

            self._send_json(200, {"filename": filename, "content": content})
            return

        self._send_json(404, {"error": "Unknown route."})

    def log_message(self, format: str, *args) -> None:
        return
