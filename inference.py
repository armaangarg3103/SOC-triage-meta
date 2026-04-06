"""
Inference Client — SOC Alert Triage

HTTP client that runs an LLM agent against a deployed environment
(local or Hugging Face Space).

Usage:
  ENV_URL=http://localhost:7860 GROQ_API_KEY=gsk_... python inference.py
  ENV_URL=https://USERNAME-soc-alert-triage.hf.space python inference.py [--task all] [--episodes 10]
"""

import argparse
import json
import os
import statistics
import sys
import time

from dotenv import load_dotenv
load_dotenv()

import httpx
from openai import OpenAI

from models import SOCAlertAction


# ---------------------------------------------------------------------------
# LLM Inference Client
# ---------------------------------------------------------------------------

def get_llm_client() -> OpenAI:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Missing GROQ_API_KEY environment variable. The inference agent requires this.", file=sys.stderr)
        sys.exit(1)
        
    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

def infer_action(obs: dict, task_id: str, client: OpenAI) -> dict:
    """
    Uses the LLM to inspect the given observation and formulate a SOCAlertAction response.
    """
    model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    
    # We use our pydantic model schema to tell the LLM exactly what to return.
    schema = SOCAlertAction.model_json_schema()
    
    system_prompt = (
        f"You are an expert Security Operations Center (SOC) Analyst AI. "
        f"Your current task is '{task_id}'.\n\n"
        f"You must analyze the provided alert observation and return your final action "
        f"as a strictly formatted JSON object that complies with the following JSON schema:\n"
        f"```json\n{json.dumps(schema)}\n```\n\n"
        f"Do not include any other text, reasoning, or markdown formatting outside of the JSON object. "
        f"For fields irrelevant to this specific task, you may omit them or return null. "
        f"Return ONLY valid, parseable JSON."
    )
    
    user_prompt = f"Observation:\n{json.dumps(obs, indent=2)}\n\n"
    if obs.get("analyst_prompt"):
        user_prompt += f"Instructions: {obs['analyst_prompt']}\n"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"⚠️ LLM Inference failed: {e}")
        # fallback to a safe empty action
        return SOCAlertAction().model_dump(exclude_none=True)


# ---------------------------------------------------------------------------
# HTTP episode runner
# ---------------------------------------------------------------------------

def run_episode(http_client: httpx.Client, llm_client: OpenAI, base_url: str, task_id: str) -> float:
    # Reset
    r = http_client.post(f"{base_url}/reset_task", json={"task_id": task_id})
    r.raise_for_status()
    obs = r.json()
    ep = obs["episode_id"]

    if task_id in ["task1_classification", "task3_response"]:
        action = infer_action(obs, task_id, llm_client)
        r2 = http_client.post(f"{base_url}/step_task?episode_id={ep}", json=action)
        r2.raise_for_status()

    elif task_id == "task2_investigation":
        while not obs.get("done"):
            action = infer_action(obs, task_id, llm_client)
            r2 = http_client.post(f"{base_url}/step_task?episode_id={ep}", json=action)
            r2.raise_for_status()
            obs = r2.json()

    # Grade
    r3 = http_client.post(f"{base_url}/grader", json={"episode_id": ep})
    r3.raise_for_status()
    result = r3.json()
    return result["final_score"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SOC Alert Triage — LLM Inference Client")
    parser.add_argument("--task", choices=["task1_classification","task2_investigation","task3_response","all"], default="all")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per task")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    base_url = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")
    llm_client = get_llm_client()

    # Health check
    try:
        r = httpx.get(f"{base_url}/health", timeout=10)
        r.raise_for_status()
        print(f"✅ Connected to environment at {base_url}")
    except Exception as e:
        print(f"❌ Cannot reach environment at {base_url}: {e}")
        sys.exit(1)

    tasks = (
        ["task1_classification", "task2_investigation", "task3_response"]
        if args.task == "all" else [args.task]
    )

    all_results = {}

    with httpx.Client(timeout=60) as http_client:
        print("\n" + "=" * 60)
        print("  SOC Alert Triage — LLM Inference Results")
        print("=" * 60)

        for task_id in tasks:
            scores = []
            print(f"\n[{task_id}] Running {args.episodes} episodes...")
            for ep_num in range(args.episodes):
                try:
                    start = time.time()
                    score = run_episode(http_client, llm_client, base_url, task_id)
                    elapsed = time.time() - start
                    scores.append(score)
                    if args.verbose:
                        print(f"  Episode {ep_num+1:3d}: {score:.3f}  ({elapsed:.2f}s)")
                except Exception as e:
                    print(f"  Episode {ep_num+1}: ERROR — {e}")

            if scores:
                mean   = statistics.mean(scores)
                stdev  = statistics.stdev(scores) if len(scores) > 1 else 0.0
                all_results[task_id] = mean
                print(f"  Mean: {mean:.4f}  StdDev: {stdev:.4f}  Min: {min(scores):.4f}  Max: {max(scores):.4f}")
            else:
                all_results[task_id] = 0.0
                print("  No successful episodes.")

    print("\n" + "=" * 60)
    print("  Final Summary — copy these into openenv.yaml + README")
    print("=" * 60)
    for task_id, score in all_results.items():
        print(f"  {task_id:<28}: {score:.4f}")
    print()
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
