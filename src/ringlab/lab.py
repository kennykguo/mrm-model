"""A bounded, loopback-only browser lab over the real simulation API.

The browser supplies six controller/experiment scalars. It cannot choose a
physical profile, import a controller, change the solver, or write server files.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import platform
import socket
import threading

import numpy as np
import yaml

from .config import load
from .experiments import plain, versions
from .plant import simulate
from .provenance import loaded_source_hashes, require_unchanged_sources

MODEL_ID = "synthetic-single-ring-v1"
PROFILE_PATH = Path(__file__).with_name("lab_model.yaml")
STATIC_PATH = Path(__file__).with_name("lab_static")
PHYSICAL_SECTIONS = ("array", "ring", "thermal", "junction", "heater", "sensor", "laser", "data")
DEFAULTS = {"target_code": 260, "initial_code": 2660, "step_codes": 4,
            "deadband_codes": 2, "warmup_ms": 0.5, "ambient_step_k": 0.25}
LIMITS = {"target_code": {"min": 1, "max": 4094, "step": 1},
          "initial_code": {"min": 0, "max": 4095, "step": 1},
          "step_codes": {"min": 1, "max": 64, "step": 1},
          "deadband_codes": {"min": 0, "max": 50, "step": 1},
          "warmup_ms": {"min": 0.1, "max": 1.0, "step": 0.001},
          "ambient_step_k": {"min": -1.0, "max": 1.0, "step": 0.01}}
MAX_BODY_BYTES = 4096
MAX_CONNECTIONS = 8
ASSUMPTIONS = [
    "Synthetic single-ring teaching model; no measured private device geometry or transient validation.",
    "The step controller stabilizes a known local negative-detuning fringe; it does not acquire a resonance or identify the correct wavelength.",
    "Initial temperatures are 300 K. The initial heater code is already applied, then temperatures evolve during warm-up.",
    "Stationary NRZ photon-memory averaging uses ideal electrical edges and independent bits; this is not a bit-resolved waveform or BER prediction.",
    "The detector follows package temperature. Thermal capacities, conductances, geometry, and most device parameters are assumptions requiring calibration.",
    "Both comparisons use the same physical parameters, initial state, sensor-noise seed, and ambient disturbance.",
    "The 3D view is a schematic driven by recorded state; it is not a spatial field solution or a device layout.",
]


def _digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def physical_model(configuration):
    """Only physical parameters and schema version define the model identity."""
    resolved = load(configuration).resolved()
    return plain({"version": resolved["version"], **{key: resolved[key] for key in PHYSICAL_SECTIONS}})


def model_sha256(configuration):
    return _digest(physical_model(configuration))


def validate_controls(payload):
    if not isinstance(payload, dict) or set(payload) != set(DEFAULTS):
        raise ValueError("provide exactly: " + ", ".join(DEFAULTS))
    values = {}
    for key, bounds in LIMITS.items():
        value = payload[key]
        integer = key not in ("warmup_ms", "ambient_step_k")
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or (integer and not isinstance(value, int))):
            raise ValueError(f"{key} must be {'an integer' if integer else 'a finite number'}")
        # Test the bounded interval first so enormous JSON integers never need
        # conversion to float (which could overflow rather than return 400).
        if not bounds["min"] <= value <= bounds["max"] or not math.isfinite(value):
            raise ValueError(f"{key} must be between {bounds['min']} and {bounds['max']}")
        if key == "warmup_ms" and not math.isclose(value*1000, round(value*1000), abs_tol=1e-8):
            raise ValueError("warmup_ms must be a multiple of 0.001 ms (one simulation tick)")
        values[key] = int(value) if integer else float(value)
    return values


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def decode_controls(body):
    def nonfinite(_):
        raise ValueError("JSON numbers must be finite")
    try:
        return validate_controls(json.loads(body, object_pairs_hook=_unique_object, parse_constant=nonfinite))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("request body must be a JSON object with six controls") from error


def _warmup_s(configuration, controls):
    # Match the scheduler's representation at equality (for example 515 ticks
    # differs by one float ULP from a literal 0.000515 seconds).
    return configuration.ticks(controls["warmup_ms"]*1e-3, "warmup")*configuration.simulation.tick_s


class LabState:
    """One profile snapshot and at most one paired simulation per server."""

    def __init__(self):
        profile_bytes = PROFILE_PATH.read_bytes()
        raw = yaml.safe_load(profile_bytes)
        self._profile = load(raw)
        resolved = self._profile.resolved()
        # Refuse a future new hidden default instead of silently changing this
        # versioned profile's physical or numerical behavior.
        for section in (*PHYSICAL_SECTIONS, "simulation", "scenario"):
            if set(raw[section]) != set(resolved[section]):
                raise ValueError(f"bundled lab profile must fully resolve {section}")
        if self._profile.name != MODEL_ID or self._profile.array.count != 1:
            raise ValueError("invalid bundled single-ring lab profile")
        sources = loaded_source_hashes()
        self._metadata = {
            "model_id": MODEL_ID, "model_sha256": model_sha256(self._profile),
            "implementation_sha256": _digest(sources), "model": physical_model(self._profile),
            "defaults": copy.deepcopy(DEFAULTS), "limits": copy.deepcopy(LIMITS),
            "experiment": self._experiment(self._profile, DEFAULTS),
            "provenance": {
                "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
                "source_sha256": sources,
                "environment": {"python": platform.python_version(), "platform": platform.platform(), **versions()},
                "evidence": {key: value for key, value in resolved["evidence"].items()
                             if key.split(".", 1)[0] in PHYSICAL_SECTIONS},
                "model_hash_scope": "Canonical resolved physical sections and configuration schema version; excludes controller, experiment, and evidence labels.",
                "implementation_hash_scope": "Canonical mapping of the Python source and build/dependency file hashes captured when the package loaded; environment versions are reported separately.",
                "profile_policy": "Bundled profile is frozen at server startup; the API cannot edit or replace it.",
            },
            "assumptions": list(ASSUMPTIONS),
        }
        self.run_lock = threading.Lock()

    @staticmethod
    def _experiment(configuration, controls):
        return {"duration_s": configuration.simulation.duration_s,
                "disturbance_time_s": configuration.scenario.start_s,
                "warmup_s": _warmup_s(configuration, controls),
                "initial_temperature_k": configuration.simulation.initial_temperature_k,
                "seed": configuration.simulation.seed,
                "noise_enabled": configuration.sensor.noise_enabled,
                "simulation": asdict(configuration.simulation),
                "scenario": asdict(configuration.scenario)}

    def metadata(self):
        return copy.deepcopy(self._metadata)

    def configuration(self, controls, controller_name):
        controls = validate_controls(controls)
        if controller_name not in ("step", "hold"):
            raise ValueError("lab supports only step and hold")
        parameters = ({"target_code": controls["target_code"], "initial_code": controls["initial_code"],
                       "step_codes": controls["step_codes"], "deadband_codes": controls["deadband_codes"],
                       "warmup_s": _warmup_s(self._profile, controls)} if controller_name == "step"
                      else {"code": controls["initial_code"]})
        return load(self._profile, {"controller.name": controller_name, "controller.parameters": parameters,
                                   "simulation.initial_heater_code": controls["initial_code"],
                                   "scenario.ambient_step_k": controls["ambient_step_k"]})

    @staticmethod
    def _run_result(result, controls):
        trace = result.trace
        post = trace["sample_time_s"] >= result.configuration.scenario.start_s
        error = trace["monitor_code"][:, 0].astype(float) - controls["target_code"]
        metrics = dict(result.metrics)
        metrics["control"] = {
            "post_disturbance_rms_error_code": float(np.sqrt(np.mean(error[post]**2))) if np.any(post) else None,
            "post_disturbance_mean_abs_error_code": float(np.mean(abs(error[post]))) if np.any(post) else None,
            "final_error_code": float(error[-1]),
            "within_deadband_fraction": float(np.mean(abs(error[post]) <= controls["deadband_codes"])) if np.any(post) else None,
            "note": "Recorded released ADC readings whose physical sample time is at or after the disturbance; target-band occupancy does not prove lock or acquisition.",
        }
        warnings = []
        active = trace["time_s"] >= _warmup_s(result.configuration, controls)
        if np.any(trace["detuning_hz"][active] >= 0):
            warnings.append("Recorded detuning leaves the assumed negative-detuning fringe after warm-up; local stabilization is not guaranteed.")
        if np.any(trace["detuning_hz"][-1] >= 0):
            warnings.append("The final state is outside the controller's assumed negative-detuning fringe.")
        if np.any(np.asarray(metrics["heater_rail_time_s"]) > 0):
            warnings.append("The heater DAC reached a code rail during the experiment.")
        if np.any(np.asarray(result.diagnostics.get("sensor_clipped_samples", [])) > 0):
            warnings.append("At least one physical ADC sample clipped; clipped readings are not used for step feedback.")
        diagnostics = {**result.diagnostics, "warnings": warnings}
        return plain({"trace": trace, "metrics": metrics, "diagnostics": diagnostics,
                      "controller": result.controller_spec})

    def run(self, payload):
        controls = validate_controls(payload)
        # These checks prevent a running process from attributing old loaded
        # Python code to a new source tree after a developer edits a module.
        require_unchanged_sources()
        configurations = {name: self.configuration(controls, name) for name in ("step", "hold")}
        runs = {name: self._run_result(simulate(configuration), controls)
                for name, configuration in configurations.items()}
        require_unchanged_sources()
        response = self.metadata()
        response.update(controls=controls, experiment=self._experiment(configurations["step"], controls), runs=runs)
        return plain(response)


class LabHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = MAX_CONNECTIONS

    def __init__(self, port=8765):
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("port must be an integer between 0 and 65535")
        self.lab = LabState()
        self._connections = threading.BoundedSemaphore(MAX_CONNECTIONS)
        super().__init__(("127.0.0.1", port), LabHandler)

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(10)
        return connection, address

    def process_request(self, request, client_address):
        if not self._connections.acquire(blocking=False):
            try:
                request.sendall(b"HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connections.release()


class LabHandler(BaseHTTPRequestHandler):
    server_version = "RingLab"
    sys_version = ""

    def log_message(self, format, *args):
        # The local UI reports failures; do not echo attacker-controlled paths.
        pass

    def _respond(self, status, body, content_type="application/json; charset=utf-8"):
        if not isinstance(body, bytes):
            body = json.dumps(plain(body), allow_nan=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def send_error(self, code, message=None, explain=None):
        self._respond(code, {"error": "invalid HTTP request"})

    def _local_request(self):
        port = self.server.server_port
        hosts = self.headers.get_all("Host", [])
        origins = self.headers.get_all("Origin", [])
        if len(hosts) != 1 or hosts[0] not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            self._respond(403, {"error": "Host must identify this loopback server"})
            return False
        if len(origins) > 1 or (origins and origins[0] != f"http://{hosts[0]}"):
            self._respond(403, {"error": "cross-origin requests are not allowed"})
            return False
        if self.headers.get("Sec-Fetch-Site", "none") not in ("same-origin", "none"):
            self._respond(403, {"error": "cross-site requests are not allowed"})
            return False
        return True

    def do_GET(self):
        if not self._local_request():
            return
        if self.path == "/api/model":
            self._respond(200, self.server.lab.metadata())
            return
        assets = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/index.html": ("index.html", "text/html; charset=utf-8"),
                  "/style.css": ("style.css", "text/css; charset=utf-8"),
                  "/app.js": ("app.js", "text/javascript; charset=utf-8")}
        if self.path not in assets:
            self._respond(404, {"error": "route not found"})
            return
        filename, content_type = assets[self.path]
        try:
            body = (STATIC_PATH / filename).read_bytes()
        except OSError:
            self._respond(500, {"error": "bundled lab asset is unavailable"})
            return
        self._respond(200, body, content_type)

    def do_POST(self):
        if not self._local_request():
            return
        if self.path != "/api/run":
            self._respond(404, {"error": "route not found"})
            return
        if self.headers.get_content_type() != "application/json":
            self._respond(400, {"error": "Content-Type must be application/json"})
            return
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") or len(lengths) != 1:
            self._respond(400, {"error": "provide one bounded Content-Length; streaming bodies are not supported"})
            return
        try:
            length = int(lengths[0])
        except ValueError:
            length = -1
        if not 1 <= length <= MAX_BODY_BYTES:
            self._respond(400, {"error": f"request body must contain 1–{MAX_BODY_BYTES} bytes"})
            return
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("incomplete request body")
            controls = decode_controls(body)
        except (ValueError, socket.timeout) as error:
            self._respond(400, {"error": str(error) if isinstance(error, ValueError) else "request body timed out"})
            return
        if not self.server.lab.run_lock.acquire(blocking=False):
            self._respond(409, {"error": "a simulation is already running; wait for it to finish"})
            return
        try:
            response = self.server.lab.run(controls)
        except Exception:
            self._respond(500, {"error": "simulation could not complete; check the local process and restart if its source changed"})
        else:
            self._respond(200, response)
        finally:
            self.server.lab.run_lock.release()

    def _unsupported(self):
        if self._local_request():
            self._respond(405, {"error": "method not supported"})

    do_HEAD = do_OPTIONS = do_PUT = do_PATCH = do_DELETE = _unsupported


def serve(port=8765):
    with LabHTTPServer(port) as server:
        print(f"RingLab local control lab: http://127.0.0.1:{server.server_port}", flush=True)
        print("Fixed synthetic model; Ctrl+C stops the server.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0
