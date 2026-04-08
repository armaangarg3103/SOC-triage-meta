"""
FastAPI Application — SOC Alert Triage Environment

Endpoints:
  GET  /health
  GET  /info
  GET  /tasks
  POST /reset
  POST /step
  POST /grade
  GET  /ui        ← Gradio interactive dashboard (mounted here)

Run locally:
  python -m server.app

Docker:
  uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import os
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional

import gradio as gr
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from models import (
    EpisodeResult,
    SOCAlertAction,
    SOCAlertObservation,
    TaskID,
)
from server.environment import SOCAlertEnvironment
from server.tasks import task1_classification, task2_investigation, task3_response

# ---------------------------------------------------------------------------
# Load env vars
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# In-memory episode store  (keyed by episode_id)
# In production, swap for Redis or similar.
# ---------------------------------------------------------------------------
_EPISODES: Dict[str, SOCAlertEnvironment] = {}

# ---------------------------------------------------------------------------
# Request/Response models for HTTP layer
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id:    str
    episode_id: Optional[str] = None
    alert_id:   Optional[str] = None   # pin a specific scenario (for reproducible evals)


class GraderRequest(BaseModel):
    episode_id: str


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load data on startup
    task1_classification._load_alerts()
    task2_investigation._load_investigations()
    task3_response._load_alerts()
    yield

app = FastAPI(
    title="SOC Alert Triage Environment",
    description=(
        "OpenEnv-compatible benchmark environment for evaluating AI agents on "
        "Security Operations Centre (SOC) alert triage tasks. "
        "Three tasks: alert classification (easy), MITRE ATT&CK investigation (medium), "
        "and incident response (hard)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Health & Info endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok", "environment": "soc-alert-triage"}


@app.get("/info", tags=["meta"])
async def info():
    return {
        "name":        "SOC Alert Triage",
        "version":     "1.0.0",
        "description": "Benchmark environment for AI agent cybersecurity triage",
        "tasks":       [
            task1_classification.TASK_INFO,
            task2_investigation.TASK_INFO,
            task3_response.TASK_INFO,
        ],
        "endpoints": {
            "reset":  "POST /reset",
            "step":   "POST /step",
            "state":  "GET /state",
            "grade":  "POST /grade",
            "ui":     "GET /",
        },
    }


@app.get("/tasks", tags=["meta"])
async def list_tasks():
    t1 = dict(task1_classification.TASK_INFO)
    t1["total_scenarios"] = len(task1_classification._load_alerts())
    t2 = dict(task2_investigation.TASK_INFO)
    t2["total_scenarios"] = len(task2_investigation._load_investigations())
    t3 = dict(task3_response.TASK_INFO)
    t3["total_scenarios"] = len(task3_response._load_alerts())
    return {"tasks": [t1, t2, t3]}


# ---------------------------------------------------------------------------
# Core environment endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=SOCAlertObservation, tags=["environment"])
async def reset_task(req: ResetRequest):
    """Start a new episode. Returns the first observation."""
    try:
        env = SOCAlertEnvironment()
        obs = env.reset(
            task_id=req.task_id,
            episode_id=req.episode_id,
            alert_id=req.alert_id,
        )
        _EPISODES[obs.episode_id] = env
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=SOCAlertObservation, tags=["environment"])
async def step_task(action: SOCAlertAction, episode_id: str):
    """Submit an action and receive the next observation."""
    env = _EPISODES.get(episode_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Episode '{episode_id}' not found. Call /reset first.",
        )
    try:
        obs = env.step(action)
        return obs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state", tags=["environment"])
async def get_state(episode_id: str):
    """Return the current internal state of the episode."""
    env = _EPISODES.get(episode_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Episode '{episode_id}' not found.",
        )
    return env.state()


@app.post("/grade", response_model=EpisodeResult, tags=["environment"])
async def grade_episode(req: GraderRequest):
    """Return the full scored result for a completed episode."""
    env = _EPISODES.get(req.episode_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Episode '{req.episode_id}' not found.",
        )
    try:
        result = env.grade_episode()
        # Optionally clean up after grading
        # del _EPISODES[req.episode_id]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Gradio UI — mounted at /ui
# ---------------------------------------------------------------------------

def _build_gradio_ui() -> gr.Blocks:
    css = """
    body { font-family: 'Inter', sans-serif; }
    .soc-header { 
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 24px; border-radius: 12px; margin-bottom: 16px;
        color: white; text-align: center;
    }
    .score-box { 
        font-size: 2rem; font-weight: bold; color: #00ff88;
        text-align: center; padding: 16px; 
        background: rgba(0,255,136,0.1); border-radius: 8px;
    }
    """

    with gr.Blocks(
        title="SOC Alert Triage — OpenEnv",
        theme=gr.themes.Base(
            primary_hue="cyan",
            neutral_hue="slate",
        ),
        css=css,
    ) as demo:

        # ── Header ─────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="soc-header">
            <h1>🛡️ SOC Alert Triage Environment</h1>
            <p>Meta × Hugging Face OpenEnv Hackathon  |  Cybersecurity Benchmark</p>
            <p style="font-size:0.85rem; opacity:0.7;">
                3 Tasks: Alert Classification (Easy) → MITRE Investigation (Medium) → Incident Response (Hard)
            </p>
        </div>
        """)

        # ── Playground Tab (Standard OpenEnv layout) ────────────────────────
        with gr.Tab("▶️ Playground"):
            with gr.Row():
                # Left Column: Quick Start
                with gr.Column(scale=1):
                    gr.Markdown("""
### Quick Start
▼ **Connect to this environment**

Connect from Python using `httpx`:
```python
import httpx
env_url = "http://localhost:7860"

# Reset
res = httpx.post(f"{env_url}/reset", json={"task_id": "task1_classification"})
ep_id = res.json()["episode_id"]

# Step
action = {"is_real_alert": True, "alert_type": "phishing", "confidence": 0.9}
obs = httpx.post(f"{env_url}/step?episode_id={ep_id}", json=action)
```

▼ **Or connect directly via inference baseline script:**
```bash
python inference.py --task all
```
                    """)
                
                # Right Column: Standard Playground
                with gr.Column(scale=2):
                    gr.Markdown("### Playground\nClick **Reset** to start a new episode.")
                    pg_action = gr.Textbox(label="Message / Action (JSON)", lines=4, placeholder='{\n  "is_real_alert": true,\n  "alert_type": "phishing",\n  "confidence": 0.9\n}')
                    
                    with gr.Row():
                        pg_step_btn  = gr.Button("Step")
                        pg_reset_btn = gr.Button("Reset")
                        pg_state_btn = gr.Button("Get State")
                    
                    pg_status = gr.Textbox(label="Status", interactive=False)
                    pg_output = gr.JSON(label="Raw JSON response")

        # ── Task 1 tab ──────────────────────────────────────────────────────
        with gr.Tab("🔍 Task 1 — Alert Classification"):
            gr.Markdown("### Alert Classification\n*Easy* — Classify the alert type and decide if it's a real threat.")

            with gr.Row():
                t1_reset_btn = gr.Button("🎲 Get New Alert", variant="primary")
                t1_ep_id     = gr.Textbox(label="Episode ID", interactive=False)

            t1_alert     = gr.Textbox(label="Alert", lines=6, interactive=False)
            t1_source    = gr.Textbox(label="Source", interactive=False)
            t1_timestamp = gr.Textbox(label="Timestamp", interactive=False)
            t1_raw_log   = gr.Textbox(label="Raw Log", lines=3, interactive=False)

            gr.Markdown("---\n### 👁️ Scenario Explorer")
            t1_reveal_btn = gr.Button("👁️ Reveal Ground Truth Answers", variant="secondary")

            gr.Markdown("---\n### 📊 Answer Key")
            t1_truth    = gr.JSON(label="Ground Truth Answers")

        # ── Task 2 tab ──────────────────────────────────────────────────────
        with gr.Tab("🔬 Task 2 — MITRE Investigation"):
            gr.Markdown("### Multi-Turn MITRE ATT&CK Investigation\n*Medium* — Over 3–4 turns, identify MITRE tactic, severity, and attack timeline.")

            with gr.Row():
                t2_reset_btn = gr.Button("🎲 Start Investigation", variant="primary")
                t2_ep_id     = gr.Textbox(label="Episode ID", interactive=False)
                t2_turn_info = gr.Textbox(label="Turn", interactive=False)

            t2_alert     = gr.Textbox(label="Alert / New Context", lines=5, interactive=False)
            t2_context   = gr.Textbox(label="Additional Context", lines=4, interactive=False)
            t2_conv_hist = gr.JSON(label="Conversation History")
            t2_prompt    = gr.Textbox(label="Analyst Prompt", interactive=False)

            gr.Markdown("---\n### 👁️ Scenario Explorer")
            t2_reveal_btn = gr.Button("👁️ Reveal Ground Truth Answers", variant="secondary")

            gr.Markdown("---\n### 📊 Answer Key")
            t2_truth    = gr.JSON(label="Ground Truth Answers")

        # ── Task 3 tab ──────────────────────────────────────────────────────
        with gr.Tab("🚨 Task 3 — Incident Response"):
            gr.Markdown("### Incident Response Summary\n*Hard* — Write a full IR response: summary, containment steps, affected systems.")

            with gr.Row():
                t3_reset_btn = gr.Button("🎲 Get Incident", variant="primary")
                t3_ep_id     = gr.Textbox(label="Episode ID", interactive=False)

            t3_alert   = gr.Textbox(label="Confirmed Incident Alert", lines=6, interactive=False)
            t3_source  = gr.Textbox(label="Source", interactive=False)
            t3_raw_log = gr.Textbox(label="Raw Log", lines=3, interactive=False)

            gr.Markdown("---\n### 👁️ Scenario Explorer")
            t3_reveal_btn = gr.Button("👁️ Reveal Ground Truth Answers", variant="secondary")

            gr.Markdown("---\n### 📊 Answer Key")
            t3_truth    = gr.JSON(label="Ground Truth Answers")

        # ── About tab ───────────────────────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
## About This Environment

**SOC Alert Triage** is an OpenEnv benchmark for evaluating AI agents on cybersecurity tasks.

### Tasks

| Task | Difficulty | Grader | Score |
|---|---|---|---|
| Alert Classification | Easy | Deterministic accuracy | 0–1.0 |
| MITRE Investigation | Medium | MITRE lookup table | 0–1.0 |
| Incident Response | Hard | MITRE check + LLM judge | 0–1.0 |

### API Endpoints

```
POST /reset   {"task_id": "task1_classification"}
POST /step    {...action fields...}
POST /grade       {"episode_id": "..."}
```

### MITRE ATT&CK Integration

This environment uses the official MITRE ATT&CK framework as a public, citable taxonomy:
- **TA0001** — Initial Access  
- **TA0006** — Credential Access  
- **TA0008** — Lateral Movement  
- **TA0010** — Exfiltration  
- **TA0040** — Impact  

### Why SOC Triage?

Almost no public agent benchmarks exist for cybersecurity triage. 
Frontier models perform surprisingly poorly on SOC tasks — this benchmark reveals that gap.
            """)

        # ── Backend logic ───────────────────────────────────────────────────

        import httpx

        BASE_URL = os.getenv("ENV_URL", "http://localhost:7860")

        # Task 1 handlers
        def t1_reset():
            r = httpx.post(f"{BASE_URL}/reset", json={"task_id": "task1_classification"})
            obs = r.json()
            return (
                obs.get("episode_id", ""),
                obs.get("alert_text", ""),
                obs.get("alert_source", ""),
                obs.get("timestamp", ""),
                obs.get("raw_log", "") or "",
            )

        def t1_reveal(ep_id):
            if not ep_id:
                return None
            r2  = httpx.post(f"{BASE_URL}/grade", json={"episode_id": ep_id})
            result = r2.json()
            return result.get("ground_truth", {})

        t1_reset_btn.click(t1_reset, outputs=[t1_ep_id, t1_alert, t1_source, t1_timestamp, t1_raw_log])
        t1_reveal_btn.click(t1_reveal, inputs=[t1_ep_id], outputs=[t1_truth])

        # Task 2 handlers
        _t2_state: Dict = {}

        def t2_reset():
            r = httpx.post(f"{BASE_URL}/reset", json={"task_id": "task2_investigation"})
            obs = r.json()
            _t2_state["ep_id"]    = obs.get("episode_id", "")
            _t2_state["max_turn"] = obs.get("max_turns", 3)
            return (
                obs.get("episode_id", ""),
                f"Turn 1 / {obs.get('max_turns', 3)}",
                obs.get("alert_text", ""),
                obs.get("additional_context", "") or "",
                [],
                obs.get("analyst_prompt", ""),
            )

        def t2_reveal(ep_id):
            if not ep_id:
                return None
            r2  = httpx.post(f"{BASE_URL}/grade", json={"episode_id": ep_id})
            result = r2.json()
            return result.get("ground_truth", {})

        t2_reset_btn.click(t2_reset, outputs=[t2_ep_id, t2_turn_info, t2_alert, t2_context, t2_conv_hist, t2_prompt])
        t2_reveal_btn.click(t2_reveal, inputs=[t2_ep_id], outputs=[t2_truth])

        # Task 3 handlers
        def t3_reset():
            r = httpx.post(f"{BASE_URL}/reset", json={"task_id": "task3_response"})
            obs = r.json()
            return (
                obs.get("episode_id", ""),
                obs.get("alert_text", ""),
                obs.get("alert_source", ""),
                obs.get("raw_log", "") or "",
            )

        def t3_reveal(ep_id):
            if not ep_id:
                return None
            r2  = httpx.post(f"{BASE_URL}/grade", json={"episode_id": ep_id})
            result = r2.json()
            return result.get("ground_truth", {})

        t3_reset_btn.click(t3_reset, outputs=[t3_ep_id, t3_alert, t3_source, t3_raw_log])
        t3_reveal_btn.click(t3_reveal, inputs=[t3_ep_id], outputs=[t3_truth])

        # Playground handlers
        _pg_state = {"ep_id": ""}
        
        def pg_reset():
            r = httpx.post(f"{BASE_URL}/reset", json={"task_id": "task1_classification"})
            resp = r.json()
            _pg_state["ep_id"] = resp.get("episode_id", "")
            return f"Reset triggered (Task 1). ID: {_pg_state['ep_id']}", resp
            
        def pg_step(action_str):
            if not _pg_state.get("ep_id"):
                return "❌ No active episode. Click Reset.", {}
            try:
                import json
                action = json.loads(action_str)
            except Exception:
                return "❌ Action must be valid JSON.", {}
            try:
                r = httpx.post(f"{BASE_URL}/step?episode_id={_pg_state['ep_id']}", json=action)
                if r.status_code == 200:
                   resp = r.json()
                   return f"Step taken. Done: {resp.get('done')}", resp
                return f"HTTP Error {r.status_code}", r.json()
            except Exception as e:
                return f"Error: {e}", {}
                
        def pg_state():
            if not _pg_state.get("ep_id"):
                return "❌ No active episode.", {}
            try:
                r = httpx.get(f"{BASE_URL}/state?episode_id={_pg_state['ep_id']}")
                return "State retrieved.", r.json()
            except Exception as e:
                return f"Error: {e}", {}

        pg_reset_btn.click(pg_reset, outputs=[pg_status, pg_output])
        pg_step_btn.click(pg_step, inputs=[pg_action], outputs=[pg_status, pg_output])
        pg_state_btn.click(pg_state, outputs=[pg_status, pg_output])

    return demo


# Mount Gradio at /
if os.getenv("ENABLE_WEB_INTERFACE", "true").lower() != "false":
    gradio_app = _build_gradio_ui()
    app = gr.mount_gradio_app(app, gradio_app, path="/")


# ---------------------------------------------------------------------------
# Local dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)
