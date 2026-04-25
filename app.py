import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
import uvicorn


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOT_PATH = RESULTS_DIR / "training_results.png"
METRICS_PATH = RESULTS_DIR / "metrics.json"

app = FastAPI(title="OpenEnv RL Trainer API")

state = {
    "running": False,
    "last_started_at": None,
    "last_finished_at": None,
    "last_exit_code": None,
    "last_logs": "No training run yet.",
}


def _run_training_job():
    state["running"] = True
    state["last_started_at"] = datetime.utcnow().isoformat() + "Z"
    try:
        proc = subprocess.run(
            [sys.executable, "main.py"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
        )
        combined_logs = proc.stdout
        if proc.stderr:
            combined_logs += "\nErrors/Warnings:\n" + proc.stderr

        state["last_logs"] = combined_logs or "Training finished with no output."
        state["last_exit_code"] = proc.returncode
    except Exception as exc:
        state["last_logs"] = f"Training launcher failure: {exc}"
        state["last_exit_code"] = -1
    finally:
        state["last_finished_at"] = datetime.utcnow().isoformat() + "Z"
        state["running"] = False


@app.get("/")
def health():
    return {
        "service": "openenv-rl-trainer",
        "running": state["running"],
        "last_started_at": state["last_started_at"],
        "last_finished_at": state["last_finished_at"],
        "last_exit_code": state["last_exit_code"],
        "plot_available": PLOT_PATH.exists(),
        "metrics_available": METRICS_PATH.exists(),
    }


@app.post("/train")
def train():
    if state["running"]:
        raise HTTPException(status_code=409, detail="Training job is already running.")

    worker = threading.Thread(target=_run_training_job, daemon=True)
    worker.start()
    return {"message": "Training started.", "started_at": state["last_started_at"]}


@app.get("/logs")
def logs():
    return PlainTextResponse(state["last_logs"])


@app.get("/results/plot")
def get_plot():
    if not PLOT_PATH.exists():
        raise HTTPException(status_code=404, detail="Plot not found. Run /train first.")
    return FileResponse(str(PLOT_PATH), media_type="image/png")


@app.get("/results/metrics")
def get_metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=404, detail="Metrics not found. Run /train first."
        )
    return FileResponse(str(METRICS_PATH), media_type="application/json")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
