"""
Single-command launcher: starts the FastAPI backend in the background,
waits for it to be healthy, then launches the Streamlit UI in the
foreground. Satisfies the "must start with a single command" constraint
without requiring Docker.

Usage:
    python run.py
"""
import os
import subprocess
import sys
import time

import requests

API_HOST = "127.0.0.1"
API_PORT = os.environ.get("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}"


def wait_for_api(timeout_s: int = 20) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if requests.get(f"{API_URL}/health", timeout=2).ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    return False


def main():
    env = os.environ.copy()
    env.setdefault("API_BASE_URL", API_URL)

    print(f"Starting API on {API_URL} ...")
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.api:app", "--host", API_HOST, "--port", API_PORT],
        env=env,
    )
    try:
        if not wait_for_api():
            print("API did not become healthy in time; check the logs above.", file=sys.stderr)
        else:
            print("API is up. Launching UI...")

        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "ui/streamlit_app.py"],
            env=env,
        )
    finally:
        api_proc.terminate()
        try:
            api_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            api_proc.kill()


if __name__ == "__main__":
    main()
