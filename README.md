# 🛡️ SOC Alert Triage — OpenEnv Benchmark

[![Hugging Face](https://img.shields.io/badge/🤗%20HF%20Space-soc--alert--triage-blue)](https://huggingface.co/spaces/)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-green)](https://huggingface.co/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Meta × Hugging Face OpenEnv Hackathon** — Cybersecurity Track

A benchmark environment for evaluating AI agents on Security Operations Centre (SOC) alert triage tasks. Three difficulty tiers expose how frontier models struggle with real-world cybersecurity workflows.

> **Gap:** Almost no public agent benchmarks exist for cybersecurity triage. CyberSecEval covers code review, not SOC workflows. This is among the first of its kind.

---

## 🎯 Three Tasks

| Task | Difficulty | Grader | Max Score |
|---|---|---|---|
| `task1_classification` | **Easy** | Deterministic accuracy | 1.0 |
| `task2_investigation` | **Medium** | MITRE ATT&CK lookup table | 1.0 |
| `task3_response` | **Hard** | MITRE check + LLM judge | 1.0 |

### Task 1 — Alert Classification (Easy)
The agent receives a single SOC alert from SIEM/IDS/EDR/email gateway. It must:
1. Decide if the alert is a **real threat** or **false positive**
2. Classify the **alert type**: `phishing`, `malware`, `lateral_movement`, `data_exfil`, `false_positive`

**Scoring (deterministic):**
- `+0.50` — correct `is_real_alert` detection
- `+0.50` — correct `alert_type` classification (real alerts only)

### Task 2 — MITRE ATT&CK Investigation (Medium)
A multi-turn investigation (3–4 turns) where forensic context escalates each turn. The agent must:
- Identify the **MITRE ATT&CK tactic** (e.g., `TA0001` — Initial Access)
- Assign a **severity** (P1 Critical → P4 Low)
- Identify **which turn** the attack began

**Scoring (deterministic lookup table — MITRE is public documentation):**
- `+0.40` — correct MITRE tactic ID
- `+0.30` — correct severity (partial credit for ±1 tier)
- `+0.30` — correct attack start turn

### Task 3 — Incident Response (Hard)
A confirmed high-severity incident. The agent must produce:
- Executive `incident_summary` (2–4 sentences)
- Ordered `containment_steps`
- `affected_systems` list
- `escalate_to_ir` decision
- `mitre_technique` ID

**Scoring (hybrid):**
- `+0.30` — MITRE technique referenced (deterministic)
- `+0.30` — Containment quality (heuristic keyword coverage)
- `+0.20` — Correct IR escalation decision (deterministic)
- `+0.20` — LLM judge quality score (Groq, with heuristic fallback)

---

## 📊 Baseline Scores

> *Rule-based keyword-matching baseline. Updated after each deploy run.*

| Task | Baseline Score | Max Score |
|---|---|---|
| `task1_classification` | 0.72 | 1.0 |
| `task2_investigation` | 0.61 | 1.0 |
| `task3_response` | 0.54 | 1.0 |

---

## 🚀 Quick Start

### Local Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/soc-alert-triage
cd soc-alert-triage

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Configure environment
cp .env.example .env
# Edit .env: add your GROQ_API_KEY

# 4. Start the server
python -m server.app
# → http://localhost:7860
# → UI at http://localhost:7860/ui
```

### Docker Run

```bash
docker build -t soc-alert-triage .
docker run -p 7860:7860 --env-file .env soc-alert-triage
```

---

## 🔌 API Reference

All endpoints accept/return JSON. Episode flow: `reset → step (×N turns) → grader`.

### `GET /health`
```json
{"status": "ok", "environment": "soc-alert-triage"}
```

### `POST /reset_task`
```json
// Request
{"task_id": "task1_classification", "episode_id": "optional-uuid"}

// Response: SOCAlertObservation
{
  "task_id": "task1_classification",
  "episode_id": "abc-123",
  "alert_id": "alert_001",
  "alert_text": "High-confidence phishing email detected...",
  "alert_source": "email_gateway",
  "timestamp": "2024-03-15T08:23:41Z",
  "analyst_prompt": "...",
  "done": false,
  "reward": 0.0
}
```

### `POST /step_task?episode_id={episode_id}`
```json
// Request: SOCAlertAction (include fields relevant to the task)
{
  "is_real_alert": true,
  "alert_type": "phishing",
  "confidence": 0.95,
  "reasoning": "Spoofed domain and credential harvesting link detected."
}

// Response: SOCAlertObservation (with reward on final turn)
{
  "done": true,
  "reward": 1.0,
  "score_breakdown": {
    "real_alert_detection": 0.5,
    "alert_type_classification": 0.5
  },
  "feedback": "✅ Correctly identified threat (+0.5)  ✅ Correct alert type 'phishing' (+0.5)"
}
```

### `POST /grader`
```json
// Request
{"episode_id": "abc-123"}

// Response: EpisodeResult
{
  "episode_id": "abc-123",
  "task_id": "task1_classification",
  "final_score": 1.0,
  "score_breakdown": {...},
  "feedback": "...",
  "ground_truth": {...}
}
```

---

## 🏗️ Architecture

```
meta hack/
├── models.py                  ← Shared Pydantic schemas
├── baseline.py                ← Rule-based baseline (local)
├── inference.py               ← HTTP client (for deployed env)
├── Dockerfile                 ← All ports aligned to 7860
├── openenv.yaml               ← OpenEnv metadata
├── pyproject.toml             ← Dependencies
└── server/
    ├── app.py                 ← FastAPI + Gradio UI at /ui
    ├── environment.py         ← Main environment class
    ├── tasks/
    │   ├── task1_classification.py
    │   ├── task2_investigation.py
    │   └── task3_response.py
    ├── data/
    │   ├── alerts.json         ← 22+ SOC alert records
    │   └── investigations.json ← 8+ multi-turn scenarios
    └── tests/
        └── test_tasks.py       ← Pytest suite
```

---

## 🔑 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Yes | — | LLM judge API key (Task 3) |
| `API_BASE_URL` | No | Groq endpoint | Override LLM endpoint |
| `MODEL_NAME` | No | `llama3-8b-8192` | LLM model name |
| `ENV_URL` | No | `http://localhost:7860` | Target URL for `inference.py` |

---

## 🧪 Running Tests

```bash
pytest server/tests/ -v
```

---

## 🏆 Why This Scores High

| Feature | Benefit |
|---|---|
| MITRE ATT&CK lookup grader | Citable public taxonomy — "documented, not invented" |
| Deterministic Task 1 & 2 | Fully reproducible, no LLM dependency for core scoring |
| Heuristic fallback in Task 3 | Production-ready — works even without LLM API |
| Real SIEM-style alert data | Frontier models perform poorly on SOC tasks |
| 3-tier difficulty | Easy → Medium → Hard depth |
| Interactive Gradio UI | Human-accessible benchmark at `/ui` |

---

## 📜 License

MIT License. MITRE ATT&CK® is a registered trademark of The MITRE Corporation.
