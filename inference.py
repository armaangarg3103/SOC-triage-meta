"""
Inference Client — SOC Alert Triage
Compliant with OpenEnv Hackathon stdout constraints.
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

import httpx
from openai import OpenAI

from models import SOCAlertAction

# Hackathon mandatory configuration
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama3-8b-8192")
BENCHMARK    = "soc-alert-triage"
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Normalize action string without newlines to strictly conform to single-line regex rules
    action_str = action.replace("\n", " ").replace("\r", "") if action else "null"
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{float(r):.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_llm_client() -> OpenAI:
    if not API_KEY:
        print("❌ Missing HF_TOKEN, OPENAI_API_KEY, or GROQ_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)
        
    return OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL
    )


def infer_action(obs: dict, task_id: str, client: OpenAI) -> dict:
    schema = SOCAlertAction.model_json_schema()
    system_prompt = (
        f"You are an expert Security Operations Center Analyst AI. "
        f"Your current task is '{task_id}'.\n"
        f"Analyze observation and return your final action strictly formatted as JSON complying with the schema:\n"
        f"```json\n{json.dumps(schema)}\n```\n"
        f"Return ONLY valid, parseable JSON."
    )
    user_prompt = f"Observation:\n{json.dumps(obs, indent=2)}\n"
    if obs.get("analyst_prompt"):
        user_prompt += f"Instructions: {obs['analyst_prompt']}\n"
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception:
        # Fallback empty action
        return SOCAlertAction().model_dump(exclude_none=True)


def run_episode(http_client: httpx.Client, llm_client: OpenAI, task_id: str):
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    
    try:
        # 1. Reset
        r = http_client.post(f"{ENV_URL}/reset_task", json={"task_id": task_id})
        r.raise_for_status()
        obs = r.json()
        ep = obs["episode_id"]
        
        done = obs.get("done", False)
        
        while not done:
            steps_taken += 1
            action = infer_action(obs, task_id, llm_client)
            
            # 2. Step in env
            try:
                r2 = http_client.post(f"{ENV_URL}/step_task?episode_id={ep}", json=action)
                r2.raise_for_status()
                obs = r2.json()
                
                reward = float(obs.get("reward", 0.0))
                done = obs.get("done", False)
                error_msg = None
            except Exception as e:
                reward = 0.0
                done = True
                error_msg = str(e)
            
            rewards.append(reward)
            
            # 3. Log step with compact, single-line json action output
            action_str = json.dumps(action)
            log_step(step=steps_taken, action=action_str, reward=reward, done=done, error=error_msg)
            
            if done:
                break
                
        # 4. Final grade
        r3 = http_client.post(f"{ENV_URL}/grader", json={"episode_id": ep})
        r3.raise_for_status()
        result = r3.json()
        
        score = float(result.get("final_score", 0.0))
        success = score > 0.0 # Define base success threshold
        
    except Exception as e:
        print(f"[DEBUG] run_episode failed context: {e}", file=sys.stderr)
        
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["task1_classification","task2_investigation","task3_response","all"], default="all")
    args = parser.parse_args()

    llm_client = get_llm_client()

    tasks = (
        ["task1_classification", "task2_investigation", "task3_response"]
        if args.task == "all" else [args.task]
    )

    with httpx.Client(timeout=60) as http_client:
        for task_id in tasks:
            run_episode(http_client, llm_client, task_id)

if __name__ == "__main__":
    main()
