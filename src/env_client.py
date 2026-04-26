import httpx
from typing import Dict, Any
import os


class EnvironmentClient:
    """Interacts with the OpenEnv Space."""

    def __init__(self, api_url: str, api_key: str | None = None):
        self.api_url = api_url.rstrip("/")
        timeout_s = float(os.getenv("OPENENV_TIMEOUT_SECONDS", "30"))
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(timeout=timeout_s, headers=headers)

    def reset(self, task_id: str) -> Dict[str, Any]:
        """Start a new episode for the given task."""
        response = self.client.post(f"{self.api_url}/reset", json={"task_id": task_id})
        response.raise_for_status()
        return response.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Send an action to the environment."""
        try:
            response = self.client.post(f"{self.api_url}/step", json=action)
        except httpx.TimeoutException as exc:
            print("[API Error] Request timed out while calling /step")
            raise RuntimeError("Environment step timed out") from exc
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            print(f"[API Error] Status: {e.response.status_code}")
            print(f"[API Error] Response: {e.response.text}")
            raise
        return response.json()
