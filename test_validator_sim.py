"""
Simulate exactly what the hackathon validator does:
  1. POST /reset for each task
  2. POST /step with a minimal action
  3. POST /grade
  4. Check ALL numeric fields in every response for 0.0 or 1.0
"""
import httpx
import json
import sys

BASE = "https://armaangarg3103-soc-triage-meta.hf.space"

def check_value(label: str, val, errors: list):
    """Flag any numeric value that is exactly 0.0 or 1.0"""
    if isinstance(val, (int, float)):
        fv = float(val)
        if fv == 0.0:
            errors.append(f"  VIOLATION: {label} = {fv} (exactly 0.0)")
        elif fv == 1.0:
            errors.append(f"  VIOLATION: {label} = {fv} (exactly 1.0)")

def scan_dict(prefix: str, d: dict, errors: list):
    """Recursively scan a dict for 0.0 or 1.0 values"""
    for k, v in d.items():
        path = f"{prefix}.{k}"
        if isinstance(v, dict):
            scan_dict(path, v, errors)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    scan_dict(f"{path}[{i}]", item, errors)
                else:
                    check_value(f"{path}[{i}]", item, errors)
        else:
            check_value(path, v, errors)

TASKS = ["task1_classification", "task2_investigation", "task3_response"]

# Minimal actions for each task
MINIMAL_ACTIONS = {
    "task1_classification": {
        "is_real_alert": True,
        "alert_type": "phishing",
        "confidence": 0.5,
    },
    "task2_investigation": {
        "mitre_tactic": "TA0001",
        "severity": "P2",
        "attack_started_at_turn": 1,
        "confidence": 0.5,
    },
    "task3_response": {
        "incident_summary": "A phishing attack was detected.",
        "containment_steps": ["Isolate the host", "Block the IP", "Reset credentials"],
        "affected_systems": ["workstation-1"],
        "escalate_to_ir": True,
        "mitre_technique": "T1566",
        "confidence": 0.5,
    },
}

all_errors = []

with httpx.Client(timeout=30) as client:
    for task_id in TASKS:
        print(f"\n{'='*60}")
        print(f"TASK: {task_id}")
        print(f"{'='*60}")
        errors = []

        # 1. Reset
        print(f"\n--- POST /reset ---")
        r = client.post(f"{BASE}/reset", json={"task_id": task_id})
        reset_resp = r.json()
        print(f"  episode_id: {reset_resp.get('episode_id', 'N/A')}")
        print(f"  reward: {reset_resp.get('reward')}")
        print(f"  done: {reset_resp.get('done')}")
        scan_dict("reset", reset_resp, errors)

        ep_id = reset_resp["episode_id"]
        max_turns = reset_resp.get("max_turns", 1)

        # 2. Step (for multi-turn, step through all turns)
        action = MINIMAL_ACTIONS[task_id]
        for turn in range(max_turns):
            print(f"\n--- POST /step (turn {turn+1}/{max_turns}) ---")
            r2 = client.post(f"{BASE}/step?episode_id={ep_id}", json=action)
            step_resp = r2.json()
            print(f"  reward: {step_resp.get('reward')}")
            print(f"  done: {step_resp.get('done')}")
            if step_resp.get("score_breakdown"):
                print(f"  score_breakdown: {json.dumps(step_resp['score_breakdown'], indent=4)}")
            scan_dict(f"step_t{turn+1}", step_resp, errors)
            if step_resp.get("done"):
                break

        # 3. Grade
        print(f"\n--- POST /grade ---")
        r3 = client.post(f"{BASE}/grade", json={"episode_id": ep_id})
        grade_resp = r3.json()
        print(f"  final_score: {grade_resp.get('final_score')}")
        print(f"  score_breakdown: {json.dumps(grade_resp.get('score_breakdown', {}), indent=4)}")
        scan_dict("grade", grade_resp, errors)

        if errors:
            print(f"\n  *** {len(errors)} VIOLATIONS FOUND ***")
            for e in errors:
                print(e)
            all_errors.extend(errors)
        else:
            print(f"\n  All values OK for {task_id}")

print(f"\n{'='*60}")
if all_errors:
    print(f"TOTAL VIOLATIONS: {len(all_errors)}")
    for e in all_errors:
        print(e)
    sys.exit(1)
else:
    print("ALL CLEAR — no 0.0 or 1.0 values found anywhere.")
