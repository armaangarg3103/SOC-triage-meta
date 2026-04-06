"""
Inference Client — SOC Alert Triage

HTTP client that runs the baseline agent against a deployed environment
(local or Hugging Face Space).

Usage:
  ENV_URL=http://localhost:7860 python inference.py
  ENV_URL=https://USERNAME-soc-alert-triage.hf.space python inference.py [--task all] [--episodes 10]
"""

import argparse
import json
import os
import statistics
import sys
import time

import httpx


# ---------------------------------------------------------------------------
# Rule-based action builders (same logic as baseline.py but HTTP-based)
# ---------------------------------------------------------------------------

_PHISHING_KW  = {"phishing","spear","bec","spoof","impersonat","credential","oauth","wire","invoice","macro","xlsm"}
_MALWARE_KW   = {"cobalt strike","beacon","c2","ransomware","encrypt","lockbit","xmrig","miner","empire","shellcode","yara"}
_LATERAL_KW   = {"lateral","mimikatz","kerberoasting","rubeus","psexec","wmi","rdp","ntlm","kerberos","bloodhound"}
_EXFIL_KW     = {"exfil","dropbox","upload","dlp","dns tunnel","forwarding rule","protonmail","outbound","transfer"}
_FP_KW        = {"nessus","scheduled","backup","ci/cd","jenkins","authorized","red team","eicar","marketing","pentest"}

_TACTIC_MAP = {
    "phishing":         ("TA0001", "T1566"),
    "malware":          ("TA0011", "T1071"),
    "lateral_movement": ("TA0008", "T1021"),
    "data_exfil":       ("TA0010", "T1048"),
    "false_positive":   (None, None),
}

_SEVERITY_MAP = {
    "phishing": "P2", "malware": "P1",
    "lateral_movement": "P1", "data_exfil": "P2", "false_positive": "P4",
}


def _classify(text: str):
    t = text.lower()
    for kw in _FP_KW:
        if kw in t:
            return False, "false_positive"
    scores = {
        "phishing":         sum(1 for k in _PHISHING_KW if k in t),
        "malware":          sum(1 for k in _MALWARE_KW  if k in t),
        "lateral_movement": sum(1 for k in _LATERAL_KW  if k in t),
        "data_exfil":       sum(1 for k in _EXFIL_KW    if k in t),
    }
    best = max(scores, key=scores.get)
    return True, best


def _action_for_task1(obs: dict) -> dict:
    is_real, atype = _classify(obs.get("alert_text", ""))
    return {
        "is_real_alert": is_real,
        "alert_type":    atype if is_real else None,
        "confidence":    0.75,
        "reasoning":     f"Keyword-based: {atype}",
    }


def _action_for_task2(obs: dict) -> dict:
    text = obs.get("alert_text", "") + " " + (obs.get("additional_context") or "")
    is_real, atype = _classify(text)
    tactic, technique = _TACTIC_MAP.get(atype, (None, None))
    sev = _SEVERITY_MAP.get(atype, "P3")
    return {
        "mitre_tactic":           tactic,
        "mitre_technique":        technique,
        "severity":               sev,
        "attack_started_at_turn": 1 if is_real else None,
        "confidence":             0.65,
        "reasoning":              f"Keyword MITRE mapping: {atype}",
    }


_CONTAINMENT = {
    "phishing":         ["Block sender domain at email gateway","Quarantine malicious emails","Force password reset for clicked users","Enable MFA on affected accounts","Monitor for follow-on access"],
    "malware":          ["Isolate compromised host immediately","Kill malicious process and dump memory","Block C2 IPs and domains","Hunt for same IOCs across all endpoints","Engage IR team"],
    "lateral_movement": ["Isolate source host","Disable and rotate compromised credentials","Audit destination hosts for persistence","Rotate service account passwords","Enable DC enhanced logging"],
    "data_exfil":       ["Block outbound connections to exfil destination","Isolate responsible host","Determine sensitivity of exfiltrated data","Disable user account","Notify legal if regulated data involved"],
}


def _action_for_task3(obs: dict) -> dict:
    is_real, atype = _classify(obs.get("alert_text", ""))
    _, technique   = _TACTIC_MAP.get(atype, (None, None))
    sev            = _SEVERITY_MAP.get(atype, "P2")
    steps          = _CONTAINMENT.get(atype, ["Isolate affected systems","Investigate further","Notify security team"])
    return {
        "incident_summary":  f"A {atype.replace('_',' ')} incident classified as {sev}. Technique {technique}. Immediate containment required.",
        "containment_steps": steps,
        "affected_systems":  [obs.get("hostname") or "unknown"],
        "escalate_to_ir":    sev in ("P1","P2"),
        "mitre_technique":   technique,
        "confidence":        0.70,
        "reasoning":         f"Template IR for {atype}",
    }


# ---------------------------------------------------------------------------
# HTTP episode runner
# ---------------------------------------------------------------------------

def run_episode(client: httpx.Client, base_url: str, task_id: str) -> float:
    # Reset
    r   = client.post(f"{base_url}/reset_task", json={"task_id": task_id})
    r.raise_for_status()
    obs = r.json()
    ep  = obs["episode_id"]

    if task_id == "task1_classification":
        action = _action_for_task1(obs)
        r2 = client.post(f"{base_url}/step_task?episode_id={ep}", json=action)
        r2.raise_for_status()

    elif task_id == "task2_investigation":
        while not obs.get("done"):
            action = _action_for_task2(obs)
            r2 = client.post(f"{base_url}/step_task?episode_id={ep}", json=action)
            r2.raise_for_status()
            obs = r2.json()

    elif task_id == "task3_response":
        action = _action_for_task3(obs)
        r2 = client.post(f"{base_url}/step_task?episode_id={ep}", json=action)
        r2.raise_for_status()

    # Grade
    r3 = client.post(f"{base_url}/grader", json={"episode_id": ep})
    r3.raise_for_status()
    result = r3.json()
    return result["final_score"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SOC Alert Triage — HTTP Inference Client")
    parser.add_argument("--task", choices=["task1_classification","task2_investigation","task3_response","all"], default="all")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per task")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    base_url = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")

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

    with httpx.Client(timeout=60) as client:
        print("\n" + "=" * 60)
        print("  SOC Alert Triage — HTTP Inference Results")
        print("=" * 60)

        for task_id in tasks:
            scores = []
            print(f"\n[{task_id}] Running {args.episodes} episodes...")
            for ep_num in range(args.episodes):
                try:
                    start = time.time()
                    score = run_episode(client, base_url, task_id)
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
