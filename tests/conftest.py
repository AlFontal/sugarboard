from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import pytest
import requests


RUN_E2E = os.environ.get("RUN_E2E", "0").lower() in {"1", "true", "yes"}


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: mark as end-to-end test")


def _wait_for_health(url: str, timeout: float = 25.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(url, timeout=1.0)
            if response.status_code == 200:
                return
        except requests.RequestException:
            time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for NiceGUI health endpoint at {url}")


@pytest.fixture(scope="session")
def nicegui_server() -> Iterator[str]:
    if not RUN_E2E:
        pytest.skip("Set RUN_E2E=1 to run Selenium e2e tests")

    port = int(os.environ.get("E2E_APP_PORT", "8090"))
    data_dir = Path(__file__).parent / "data"

    env = os.environ.copy()
    env.setdefault("PORT", str(port))
    env.setdefault("SUGARBOARD_TEST_DATA_DIR", str(data_dir))
    env.setdefault("STORAGE_SECRET", "test-secret")
    env.setdefault("NICEGUI_RELOAD", "0")

    cmd = [sys.executable, "nicegui_app.py"]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_for_health(f"http://localhost:{port}/health")
        yield f"http://localhost:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
