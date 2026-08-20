"""
capture_environment.py

Captures and records the controlled environment specification, per the
project proposal (Section 2): "Record OS, CPU, RAM, Python version, SDK/API
version, browser (if applicable), internet connection, timestamp, location,
endpoint, model identifier, and configuration."

This is run ONCE per major work session (e.g., at the start of each data
collection day) to create a timestamped record of exactly what environment
the runs were executed under - so any run-to-run inconsistency can later be
checked against whether the environment itself changed (a different Python
version, a different machine, a different SDK version, etc.), which is what
"holding controlled variables constant" actually means in practice.

OUTPUT: 05_Logs_Results/environment_specs/env_spec_<timestamp>.json

Run with the VS Code Run button:
    python 03_Code/capture_environment.py
"""

import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "05_Logs_Results", "environment_specs")


def _get_package_version(pypi_package_name):
    """
    Looks up an installed package's version by its PyPI distribution name
    (e.g. "google-genai", "python-dotenv") using importlib.metadata, which
    reads the package's installed metadata directly - this is more reliable
    than trying to import the module and read a __version__ attribute,
    since many packages either don't expose one, or (for dotted-name
    packages like "google.genai") a plain __import__() call returns the
    wrong object entirely and can never find it.
    """
    try:
        import importlib.metadata
        return importlib.metadata.version(pypi_package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"
    except Exception:  # noqa: BLE001
        return "unknown"


def _get_cpu_info():
    try:
        return platform.processor() or platform.machine()
    except Exception:  # noqa: BLE001
        return "unknown"


def _get_ram_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        return "psutil not installed - run 'pip install psutil' for RAM detection"


def _get_internet_check():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return "connected"
    except OSError:
        return "no connection detected"


def capture():
    spec = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "cpu": _get_cpu_info(),
            "ram_gb": _get_ram_gb(),
            "machine_type": platform.machine(),
        },
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "sdk_versions": {
            "google-genai": _get_package_version("google-genai"),
            "mistralai": _get_package_version("mistralai"),
            "groq": _get_package_version("groq"),
            "openai": _get_package_version("openai"),
            "python-dotenv": _get_package_version("python-dotenv"),
        },
        "network": {
            "internet_status": _get_internet_check(),
            "hostname": socket.gethostname(),
        },
        "model_endpoints": {
            "gemini": {"model": "gemini-3.5-flash-lite", "temperature": 0.7},
            "mistral": {"model": "open-mistral-nemo", "temperature": 0.7},
            "groq": {"model": "openai/gpt-oss-120b", "temperature": 0.7},
        },
    }
    return spec


def main():
    spec = capture()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Use a clean, readable filename format instead of the raw ISO timestamp
    # (which produces messy strings like "...T09-58-00.986853+00-00.json")
    from datetime import datetime as _dt
    now = _dt.now(timezone.utc)
    filename = f"env_spec_{now.strftime('%Y%m%d_%H%M%S')}.json"
    output_path = os.path.join(OUTPUT_DIR, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)

    print("Environment specification captured:\n")
    print(json.dumps(spec, indent=2))
    print(f"\nSaved to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()