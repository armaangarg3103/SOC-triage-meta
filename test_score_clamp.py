"""
Integration test: verify that clamped scores are always strictly in (0, 1).
Run from the project root: python test_score_clamp.py
"""
import sys
sys.path.insert(0, ".")

from server.tasks import task1_classification, task2_investigation, task3_response
from models import SOCAlertAction, AlertType, Severity

_SCORE_EPS = 1e-3


def _clamp_score(score: float) -> float:
    return max(_SCORE_EPS, min(1.0 - _SCORE_EPS, float(score)))


def check(label: str, raw_score: float) -> bool:
    clamped = _clamp_score(raw_score)
    ok = 0 < clamped < 1
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] {label}: raw={raw_score:.4f}  clamped={clamped:.4f}")
    return ok


all_ok = True

# ── Task 1 ──────────────────────────────────────────────────────────────────
print("=== Task 1 – Alert Classification ===")
alerts = task1_classification._load_alerts()
scenario = alerts[0]
print(f"  Scenario: {scenario['id']}  is_real={scenario['is_real_alert']}  type={scenario.get('alert_type')}")

# All-wrong action
action_wrong = SOCAlertAction(is_real_alert=(not scenario["is_real_alert"]), alert_type=AlertType.phishing)
s, bd, _ = task1_classification.grade(action_wrong, scenario)
all_ok &= check("T1 all-wrong  total", s)
for k, v in bd.items():
    all_ok &= check(f"  T1 {k}", v)

# All-correct action
try:
    correct_type = AlertType(scenario["alert_type"]) if scenario.get("alert_type") else None
except Exception:
    correct_type = None
action_right = SOCAlertAction(is_real_alert=scenario["is_real_alert"], alert_type=correct_type)
s2, bd2, _ = task1_classification.grade(action_right, scenario)
all_ok &= check("T1 all-correct total", s2)
for k, v in bd2.items():
    all_ok &= check(f"  T1 {k}", v)

# ── Task 2 ──────────────────────────────────────────────────────────────────
print("\n=== Task 2 – MITRE Investigation ===")
invs = task2_investigation._load_investigations()
inv = next((i for i in invs if i.get("attack_type") != "false_positive"), invs[0])
print(f"  Scenario: {inv['id']}  tactic={inv.get('ground_truth_mitre_tactic')}  sev={inv.get('ground_truth_severity')}")

# All-wrong
action_wrong2 = SOCAlertAction(mitre_tactic="TA9999", severity=Severity.P4, attack_started_at_turn=99)
s3, bd3, _ = task2_investigation.grade(action_wrong2, inv)
all_ok &= check("T2 all-wrong  total", s3)
for k, v in bd3.items():
    all_ok &= check(f"  T2 {k}", v)

# All-correct
correct_tactic = inv.get("ground_truth_mitre_tactic", "TA0001")
correct_sev    = inv.get("ground_truth_severity", "P1")
correct_turn   = inv.get("attack_started_at_turn", 1)
action_right2 = SOCAlertAction(
    mitre_tactic=correct_tactic,
    severity=Severity(correct_sev),
    attack_started_at_turn=correct_turn,
)
s4, bd4, _ = task2_investigation.grade(action_right2, inv)
all_ok &= check("T2 all-correct total", s4)
for k, v in bd4.items():
    all_ok &= check(f"  T2 {k}", v)

# ── Task 3 heuristic edge cases ─────────────────────────────────────────────
print("\n=== Task 3 – Incident Response (heuristic judge) ===")
t3_alerts = task3_response._load_alerts()
t3_scenario = t3_alerts[0]
print(f"  Scenario: {t3_scenario['id']}")

# Completely empty action
action_empty = SOCAlertAction(
    incident_summary=None,
    containment_steps=None,
    affected_systems=None,
    escalate_to_ir=None,
    mitre_technique=None,
)
s5, bd5, _ = task3_response.grade(action_empty, t3_scenario)
all_ok &= check("T3 empty action total", s5)
for k, v in bd5.items():
    all_ok &= check(f"  T3 {k}", v)

# Rich action (likely near max-score)
ground_tech = t3_scenario.get("ground_truth_mitre_technique", "T1566")
ground_sev_t3 = t3_scenario.get("ground_truth_severity", "P1")
should_esc = ground_sev_t3 in ("P1", "P2")
action_rich = SOCAlertAction(
    incident_summary=(
        f"A confirmed high-severity {ground_tech} attack was detected. "
        "The attacker executed a lateral movement across multiple endpoints. "
        "Immediate isolation and forensic preservation are required."
    ),
    containment_steps=[
        "Isolate affected hosts from the network immediately",
        "Block attacker IP at the firewall",
        "Disable compromised user account credentials",
        "Reset all affected user passwords and revoke active sessions",
        "Preserve forensic disk images and memory dumps before remediation",
        "Quarantine malicious email attachments in the email gateway",
    ],
    affected_systems=["WIN-SRVR01", "WORKSTATION-42"],
    escalate_to_ir=should_esc,
    mitre_technique=ground_tech,
    reasoning=f"MITRE technique {ground_tech} was identified in the raw logs.",
)
s6, bd6, _ = task3_response.grade(action_rich, t3_scenario)
all_ok &= check("T3 rich action total", s6)
for k, v in bd6.items():
    all_ok &= check(f"  T3 {k}", v)

# ── Summary ─────────────────────────────────────────────────────────────────
print()
if all_ok:
    print("All checks PASSED! Scores are strictly in (0, 1).")
else:
    print("SOME CHECKS FAILED — review output above.")
    sys.exit(1)
