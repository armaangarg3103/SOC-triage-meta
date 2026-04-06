# Prompt Injection Environment Instructions

## Purpose

Use this file when creating, extending, or debugging the environment in this repository.

This repo is an OpenEnv-style FastAPI environment for evaluating an agent that detects and sanitizes prompt injection attacks across 3 tasks:

- `task1_detection`: single-turn detection
- `task2_multiturn`: multi-turn detection
- `task3_adversarial`: adversarial sanitization

## Real Runtime Structure

These are the files that matter most at runtime:

- `server/app.py`
  Creates the FastAPI/OpenEnv app, mounts extra endpoints, and mounts the Gradio UI at `/ui`.
- `server/environment.py`
  This is the real environment implementation used by the app.
- `models.py`
  Defines the action and observation schema shared by the app, tasks, baseline, and inference.
- `server/tasks/task1_detection.py`
  Scenario generation and grading for task 1.
- `server/tasks/task2_multiturn.py`
  Scenario generation and grading for task 2.
- `server/tasks/task3_adversarial.py`
  Scenario generation and grading for task 3, including LLM-based sanitization judging.
- `server/data/*.json`
  The static benchmark data for benign inputs, attacks, multi-turn scenarios, and adversarial cases.
- `baseline.py`
  Local baseline runner against the Python environment class.
- `inference.py`
  Client-style runner that calls the HTTP endpoints of a deployed environment.
- `Dockerfile`
  Container build and runtime entrypoint.
- `openenv.yaml`
  Environment metadata and task declarations for the OpenEnv/Hugging Face-style setup.

## Important Caveat

There is also a legacy/template file:

- `server/prompt_injection_env_environment.py`

and `server/__init__.py` currently exports that class instead of the real one in `server/environment.py`.

Treat `server/environment.py` as the source of truth for actual environment behavior.

## Request/Response Contract

### Observation model

The agent sees a `PromptInjectionObservation` from `models.py` with fields such as:

- `task_id`
- `turn`
- `max_turns`
- `message`
- `context`
- `user_intent`
- `conversation_history`
- `attack_sophistication`
- `episode_id`
- `done`
- `reward`
- `score_breakdown`
- `feedback`

### Action model

The agent submits `PromptInjectionAction` with these key fields:

- `is_injection`
- `confidence`
- `attack_type`
- `attack_started_at_turn`
- `attack_sophistication`
- `sanitized_message`
- `reasoning`

If you change these models, you must keep `server/app.py`, all task graders, `baseline.py`, and `inference.py` in sync.

## How Episodes Flow

1. `POST /reset_task` calls `PromptInjectionEnvironment.reset(task_id=...)`.
2. `server/environment.py` selects one scenario builder based on task id.
3. A task-specific observation is returned to the agent.
4. `POST /step_task` validates the action against `PromptInjectionAction`.
5. The environment routes grading to the task-specific grader.
6. `POST /grader` returns final episode scoring.

Task-specific routing currently lives in `server/environment.py`.

## What To Edit For Each Kind Of Change

### If you want to add a new task

Do:

- Add a new `TaskID` enum value in `models.py`.
- Create a new task module under `server/tasks/`.
- Add scenario builder, observation builder, grader, and task info in that module.
- Wire the task into `server/environment.py` inside `reset()`, `step()`, and `grade_episode()`.
- Add the task metadata to `openenv.yaml`.
- Add tests in `server/tests/test_tasks.py`.

Do not:

- Add task logic directly inside `server/app.py`.
- Reuse task 1 or task 2 grading paths for a task with different semantics.

### If you want to change scoring

Do:

- Update the grading function in the relevant `server/tasks/task*.py` file.
- Keep `feedback` and `score_breakdown` informative, because the UI and debugging flow depend on them.
- Update expected baseline values in `openenv.yaml`, README, and task info if scores materially change.

Do not:

- Change scores in only one place and leave metadata stale.
- Remove reward breakdown fields unless every consumer is updated.

### If you want to change input data

Do:

- Edit the relevant JSON file in `server/data/`.
- Preserve the existing record shape for the task loader.
- Keep contexts aligned with the `Context` enum in `models.py`.

Do not:

- Introduce new attack type strings without updating `AttackType`.
- Add malformed records, because task builders assume required keys exist.

### If you want to change the HTTP API

Do:

- Update `server/app.py`.
- Preserve compatibility with `inference.py` unless you are intentionally versioning the API.
- Keep `reset_task`, `step_task`, and `grader` behavior stable for agent integrations.

Do not:

- Rename endpoints casually.
- Return ad hoc payload shapes that bypass the Pydantic/OpenEnv models.

### If you want to change model-based judging

Do:

- Update `task3_adversarial.py` for sanitization judging.
- Update `baseline.py`, `inference.py`, `server/attacker.py`, or `server/simulator.py` if shared LLM config behavior changes.
- Keep `.env`-driven configuration working through `GROQ_API_KEY`, `API_BASE_URL`, and `MODEL_NAME`.

Do not:

- Hardcode secrets.
- Assume network access exists during all test runs.

## Environment Variables

Expected variables:

- `GROQ_API_KEY`
- `API_BASE_URL` optional, defaults to Groq-compatible OpenAI endpoint
- `MODEL_NAME` optional
- `ENV_URL` used by `inference.py`

Docker also sets:

- `ENVIRONMENT`
- `MAX_EPISODES`
- `ENABLE_WEB_INTERFACE`

## Docker And Local Run Notes

### Local Python run

Typical flow:

1. Install dependencies from `pyproject.toml`.
2. Create `.env` with `GROQ_API_KEY`.
3. Run `python -m server.app` or equivalent app startup.

### Docker run

The container command starts:

- `uvicorn server.app:app --host 0.0.0.0 --port 7860`

Important mismatch to remember:

- `openenv.yaml` uses port `7860`
- `Dockerfile` exposes `8000`
- healthcheck calls `http://localhost:8000/health`
- runtime command serves `7860`

If you need production reliability, keep the exposed port, healthcheck port, and runtime port aligned.

## Repo-Specific Do List

- Use `server/environment.py` as the main environment implementation.
- Keep task-specific logic inside `server/tasks/`.
- Keep schema changes centralized in `models.py`.
- Keep static benchmark data under `server/data/`.
- Update metadata in `openenv.yaml` and README when benchmark behavior changes.
- Add or update tests when you change task logic or scoring.
- Preserve the distinction between local environment execution (`baseline.py`) and remote HTTP execution (`inference.py`).

## Repo-Specific Don't List

- Do not treat `server/prompt_injection_env_environment.py` as the active runtime implementation.
- Do not change enum values without updating all JSON data and prompt builders.
- Do not break the `/reset_task -> /step_task -> /grader` flow.
- Do not move grading logic into UI code.
- Do not store secrets in source files.
- Do not rely on LLM availability for every code path unless you also provide a fallback.

## Suggested Safe Workflow

1. Decide whether your change is schema, task logic, data, API, or deployment.
2. Edit the smallest relevant file set.
3. Re-check `models.py` compatibility.
4. Re-check `server/environment.py` routing.
5. Re-check `openenv.yaml` metadata if task behavior changed.
6. Run tests.
7. If task 3 changed, manually inspect the LLM-judge path and fallback behavior.

## Current Gotchas I Found

- `server/__init__.py` points to the legacy environment file, not the main runtime file.
- Test collection currently fails without `openenv` installed because of that import path.
- The Dockerfile healthcheck and exposed port do not match the actual uvicorn runtime port.
- README, `openenv.yaml`, and task modules do not all agree on expected baseline scores.

## Minimum Checklist Before Shipping Changes

- Schema still validates end to end.
- Task data matches enums.
- Environment reset and step paths still work.
- Episode grading still returns `final_score`, `feedback`, and ground truth.
- Docker port and healthcheck behavior are still coherent.
- Any score change is reflected in metadata and docs.
