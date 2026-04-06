"""
Baseline Agent — SOC Alert Triage

A rule-based baseline that runs episodes locally (no HTTP).
Used to establish reproducible baseline scores.

Usage:
  python baseline.py [--task task1_classification] [--episodes 20]

Rules:
  Task 1 — keyword matching on alert text
  Task 2 — keyword-to-MITRE-tactic lookup
  Task 3 — template-based incident summary generation
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from models import AlertType, SOCAlertAction, Severity, TaskID
from server.environment import SOCAlertEnvironment


# ---------------------------------------------------------------------------
# Keyword matching helpers
# ---------------------------------------------------------------------------

_PHISHING_KEYWORDS = {
    "phishing", "spear", "bec", "spoof", "impersonat", "credential", "harvest",
    "oauth", "consent", "wire transfer", "payment", "invoice", "bit.ly",
    "docusign", "credential", "macro", "xlsm", "vba",
}

_MALWARE_KEYWORDS = {
    "cobalt strike", "beacon", "c2", "ransomware", "encrypt", "lockbit",
    "xmrig", "miner", "cryptomin", "empire", "powershell empire",
    "dropper", "shellcode", "reflective", "yara", "conti",
}

_LATERAL_KEYWORDS = {
    "lateral", "pass-the-hash", "mimikatz", "kerberoasting", "rubeus",
    "psexec", "wmi", "rdp", "ntlm", "kerberos", "spn", "bloodhound",
    "pth", "lsass",
}

_EXFIL_KEYWORDS = {
    "exfil", "dropbox", "upload", "data loss", "dlp", "dns tunnel",
    "forwarding rule", "protonmail", "tutanota", "outbound", "transfer",
    "large.*upload", "s3", "onedrive personal",
}

_FP_KEYWORDS = {
    "nessus", "scheduled", "backup", "ci/cd", "pipeline", "jenkins",
    "authorized", "red team", "penetration test", "eicar", "marketing",
    "campaign", "pentest",
}


def _classify_alert(alert_text: str) -> tuple[bool, str]:
    """
    Simple rule-based classifier.
    Returns (is_real_alert, alert_type).
    """
    text = alert_text.lower()

    # False positive signals are strongest
    for kw in _FP_KEYWORDS:
        if kw in text:
            return False, "false_positive"

    scores = {
        "phishing":         sum(1 for k in _PHISHING_KEYWORDS if k in text),
        "malware":          sum(1 for k in _MALWARE_KEYWORDS   if k in text),
        "lateral_movement": sum(1 for k in _LATERAL_KEYWORDS   if k in text),
        "data_exfil":       sum(1 for k in _EXFIL_KEYWORDS     if k in text),
    }

    best_type  = max(scores, key=scores.get)
    best_score = scores[best_type]

    if best_score == 0:
        # No clear match — default to malware (common)
        return True, "malware"

    return True, best_type


_TACTIC_MAP = {
    "phishing":         ("TA0001", "T1566"),
    "malware":          ("TA0011", "T1071"),
    "lateral_movement": ("TA0008", "T1021"),
    "data_exfil":       ("TA0010", "T1048"),
    "false_positive":   (None, None),
}

_ALERT_SEVERITY_MAP = {
    "phishing":         "P2",
    "malware":          "P1",
    "lateral_movement": "P1",
    "data_exfil":       "P2",
    "false_positive":   "P4",
}


def _build_action_task1(obs) -> SOCAlertAction:
    is_real, alert_type = _classify_alert(obs.alert_text)
    return SOCAlertAction(
        is_real_alert=is_real,
        alert_type=AlertType(alert_type) if is_real else None,
        confidence=0.75,
        reasoning=f"Keyword-based classification: detected '{alert_type}' indicators.",
    )


def _build_action_task2(obs) -> SOCAlertAction:
    is_real, alert_type = _classify_alert(obs.alert_text + " " + (obs.additional_context or ""))
    tactic, technique = _TACTIC_MAP.get(alert_type, (None, None))
    severity          = _ALERT_SEVERITY_MAP.get(alert_type, "P3")

    # Simple heuristic for attack start turn
    # If the attack was in the first message it's turn 1, otherwise guess turn 1
    return SOCAlertAction(
        mitre_tactic=tactic,
        mitre_technique=technique,
        severity=Severity(severity) if severity != "P4" else Severity.P4,
        attack_started_at_turn=1 if is_real else None,
        confidence=0.65,
        reasoning=f"Rule-based MITRE mapping for detected '{alert_type}' pattern.",
    )


def _build_action_task3(obs) -> SOCAlertAction:
    is_real, alert_type = _classify_alert(obs.alert_text)
    _, technique        = _TACTIC_MAP.get(alert_type, (None, None))
    severity            = _ALERT_SEVERITY_MAP.get(alert_type, "P2")

    containment_templates = {
        "phishing": [
            "Block the sender domain and IP at email gateway",
            "Quarantine all copies of the malicious email",
            "Force password reset for affected users",
            "Enable MFA on all impacted accounts",
            "Alert users who clicked links and monitor for anomalous activity",
        ],
        "malware": [
            "Isolate the compromised host(s) from the network immediately",
            "Kill the malicious process and preserve memory dump",
            "Block malicious IPs and domains at firewall and DNS",
            "Run threat hunt across all endpoints for same IOCs",
            "Engage IR team — assume lateral movement has occurred",
        ],
        "lateral_movement": [
            "Isolate the source host of the lateral movement",
            "Disable and reset compromised account credentials",
            "Audit all destination hosts for persistence mechanisms",
            "Rotate all potentially exposed service account passwords",
            "Enable enhanced logging on domain controllers",
        ],
        "data_exfil": [
            "Block all outbound connections to identified exfiltration destinations",
            "Isolate the host responsible for the data transfer",
            "Determine what data was exfiltrated and its sensitivity",
            "Disable the responsible user account",
            "Engage legal counsel if regulated data (PII/PCI) may be involved",
        ],
    }

    steps = containment_templates.get(alert_type, [
        "Isolate affected systems",
        "Investigate the alert further",
        "Notify security team",
    ])

    summary = (
        f"A {alert_type.replace('_', ' ')} incident has been confirmed. "
        f"Immediate containment actions are required. "
        f"The incident has been classified as {severity} severity. "
        f"MITRE technique {technique or 'under investigation'} applies."
    )

    return SOCAlertAction(
        incident_summary=summary,
        containment_steps=steps,
        affected_systems=[obs.hostname or "unknown"],
        escalate_to_ir=severity in ("P1", "P2"),
        mitre_technique=technique,
        confidence=0.70,
        reasoning=f"Template IR response for '{alert_type}' pattern.",
    )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(task_id: str) -> float:
    env = SOCAlertEnvironment()
    obs = env.reset(task_id=task_id)

    if task_id == TaskID.task1_classification:
        action = _build_action_task1(obs)
        env.step(action)

    elif task_id == TaskID.task2_investigation:
        # Multi-turn — step through all turns
        while not obs.done:
            action = _build_action_task2(obs)
            obs    = env.step(action)

    elif task_id == TaskID.task3_response:
        action = _build_action_task3(obs)
        env.step(action)

    result = env.grade_episode()
    return result.final_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SOC Alert Triage — Baseline Agent")
    parser.add_argument(
        "--task",
        choices=["task1_classification", "task2_investigation", "task3_response", "all"],
        default="all",
        help="Which task to run baseline on",
    )
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes per task")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    tasks = (
        ["task1_classification", "task2_investigation", "task3_response"]
        if args.task == "all"
        else [args.task]
    )

    print("\n" + "=" * 60)
    print("   SOC Alert Triage — Baseline Agent Results")
    print("=" * 60)

    all_results = {}
    for task_id in tasks:
        scores = []
        print(f"\n[{task_id}] Running {args.episodes} episodes...")
        for ep in range(args.episodes):
            score = run_episode(task_id)
            scores.append(score)
            if args.verbose:
                print(f"  Episode {ep+1:3d}: {score:.3f}")

        mean   = statistics.mean(scores)
        median = statistics.median(scores)
        stdev  = statistics.stdev(scores) if len(scores) > 1 else 0.0
        all_results[task_id] = mean

        print(f"  Mean:   {mean:.4f}")
        print(f"  Median: {median:.4f}")
        print(f"  StdDev: {stdev:.4f}")
        print(f"  Min:    {min(scores):.4f}   Max: {max(scores):.4f}")

    print("\n" + "=" * 60)
    print("   Summary (update openenv.yaml + README with these)")
    print("=" * 60)
    for task_id, score in all_results.items():
        print(f"  {task_id:<28}: {score:.4f}")
    print()

    # Output machine-readable JSON to stdout for piping
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
