import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
import uvicorn


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOT_PATH = RESULTS_DIR / "training_results.png"
METRICS_PATH = RESULTS_DIR / "metrics.json"

app = FastAPI(title="OpenEnv RL Trainer API")
state_lock = threading.Lock()

state = {
    "running": False,
    "last_started_at": None,
    "last_finished_at": None,
    "last_exit_code": None,
    "last_logs": "No training run yet.",
}


def _run_training_job():
    with state_lock:
        state["running"] = True
        state["last_started_at"] = datetime.utcnow().isoformat() + "Z"
        state["last_logs"] = ""
    try:
        proc = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        combined_lines = []
        assert proc.stdout is not None
        for line in proc.stdout:
            combined_lines.append(line)
            with state_lock:
                state["last_logs"] = "".join(combined_lines)

        exit_code = proc.wait()
        with state_lock:
            if not combined_lines:
                state["last_logs"] = "Training finished with no output."
            state["last_exit_code"] = exit_code
    except Exception as exc:
        with state_lock:
            state["last_logs"] = f"Training launcher failure: {exc}"
            state["last_exit_code"] = -1
    finally:
        with state_lock:
            state["last_finished_at"] = datetime.utcnow().isoformat() + "Z"
            state["running"] = False


def _maybe_start_training() -> bool:
    with state_lock:
        if state["running"]:
            return False
    worker = threading.Thread(target=_run_training_job, daemon=True)
    worker.start()
    return True


@app.on_event("startup")
def startup():
    auto_start = os.getenv("AUTO_START_TRAINING", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if auto_start:
        _maybe_start_training()


@app.get("/")
def dashboard():
        return HTMLResponse(
                """
<!doctype html>
<html>
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
        <title>OpenEnv RL Trainer</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 24px; background: #f5f7fb; color: #1f2937; }
            .wrap { max-width: 980px; margin: 0 auto; }
            .card { background: white; border-radius: 10px; padding: 16px; box-shadow: 0 4px 16px rgba(0,0,0,0.08); margin-bottom: 16px; }
            button { background: #2563eb; color: white; border: none; padding: 10px 14px; border-radius: 8px; cursor: pointer; }
            button:disabled { background: #94a3b8; cursor: not-allowed; }
            pre { background: #0f172a; color: #e2e8f0; padding: 12px; border-radius: 8px; min-height: 260px; overflow: auto; }
            .meta { font-size: 14px; line-height: 1.6; }
            .links a { margin-right: 10px; }
        </style>
    </head>
    <body>
        <div class=\"wrap\">
            <div class=\"card\">
                <h2>OpenEnv RL Trainer</h2>
                <p>Target env: https://aaditya-7937-openenv-review.hf.space</p>
                <button id=\"trainBtn\">Start Training</button>
                <span id=\"statusText\" style=\"margin-left:10px;\"></span>
            </div>

            <div class=\"card meta\" id=\"meta\"></div>

            <div class=\"card\">
                <h3>Live Logs</h3>
                <pre id=\"logs\">No training run yet.</pre>
            </div>

            <div class=\"card links\">
                <a href=\"/results/plot\" target=\"_blank\">Open plot</a>
                <a href=\"/results/metrics\" target=\"_blank\">Open metrics</a>
            </div>
        </div>

        <script>
            async function refreshStatus() {
                const res = await fetch('/status');
                const data = await res.json();
                const meta = document.getElementById('meta');
                meta.innerHTML =
                    '<b>Running:</b> ' + data.running + '<br>' +
                    '<b>Last started:</b> ' + (data.last_started_at || '-') + '<br>' +
                    '<b>Last finished:</b> ' + (data.last_finished_at || '-') + '<br>' +
                    '<b>Last exit code:</b> ' + (data.last_exit_code ?? '-');

                const btn = document.getElementById('trainBtn');
                btn.disabled = data.running;
                document.getElementById('statusText').textContent = data.running ? 'Training in progress...' : 'Idle';
            }

            async function refreshLogs() {
                const res = await fetch('/logs');
                const txt = await res.text();
                const el = document.getElementById('logs');
                el.textContent = txt || 'No logs yet.';
                el.scrollTop = el.scrollHeight;
            }

            document.getElementById('trainBtn').addEventListener('click', async () => {
                await fetch('/train', { method: 'POST' });
                await refreshStatus();
            });

            setInterval(() => { refreshStatus(); refreshLogs(); }, 2000);
            refreshStatus();
            refreshLogs();
        </script>
    </body>
</html>
"""
        )


@app.get("/status")
def health():
    with state_lock:
        snapshot = dict(state)
    return {
        "service": "openenv-rl-trainer",
        "running": snapshot["running"],
        "last_started_at": snapshot["last_started_at"],
        "last_finished_at": snapshot["last_finished_at"],
        "last_exit_code": snapshot["last_exit_code"],
        "plot_available": PLOT_PATH.exists(),
        "metrics_available": METRICS_PATH.exists(),
    }


@app.post("/train")
def train():
    with state_lock:
        running = state["running"]
    if running:
        raise HTTPException(status_code=409, detail="Training job is already running.")

    started = _maybe_start_training()
    if not started:
        raise HTTPException(status_code=409, detail="Training job is already running.")
    with state_lock:
        started_at = state["last_started_at"]
    return {"message": "Training started.", "started_at": started_at}


@app.get("/logs")
def logs():
    with state_lock:
        current_logs = state["last_logs"]
    return PlainTextResponse(current_logs)


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
