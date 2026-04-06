"""
Task 2 — MITRE ATT&CK Investigation (Medium)

Agent receives a multi-turn SOC investigation (3–4 turns).
Each turn reveals more context about an evolving incident.

After each turn the agent must keep refining, and on the FINAL turn
it must provide:
  - mitre_tactic    : MITRE Tactic ID (e.g. "TA0001")
  - mitre_technique : MITRE Technique ID (e.g. "T1566")
  - severity        : P1 | P2 | P3 | P4
  - attack_started_at_turn : which turn the attack began

Grader: Deterministic lookup table — no LLM required.
  +0.40 for correct MITRE tactic
  +0.30 for correct severity
  +0.30 for correct attack_started_at_turn
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from models import SOCAlertAction, SOCAlertObservation, TaskID

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_DATA_FILE = Path(__file__).parent.parent / "data" / "investigations.json"
_INVESTIGATIONS: list[dict] = []


def _load_investigations() -> list[dict]:
    global _INVESTIGATIONS
    if not _INVESTIGATIONS:
        with open(_DATA_FILE, encoding="utf-8") as f:
            _INVESTIGATIONS = json.load(f)
    return _INVESTIGATIONS


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------

def build_scenario(investigation_id: Optional[str] = None) -> dict:
    investigations = _load_investigations()
    if investigation_id:
        inv = next((i for i in investigations if i["id"] == investigation_id), None)
        if inv is None:
            raise ValueError(f"Investigation '{investigation_id}' not found")
        return inv
    return random.choice(investigations)


# ---------------------------------------------------------------------------
# Observation builders (one per turn)
# ---------------------------------------------------------------------------

def build_observation(
    scenario: dict,
    episode_id: str,
    turn: int = 1,
    conversation_history: Optional[List[Dict]] = None,
) -> SOCAlertObservation:
    """Build the observation for a given turn in the investigation."""
    turn_data = scenario["turns"][turn - 1]
    max_turns = scenario["max_turns"]

    return SOCAlertObservation(
        task_id=TaskID.task2_investigation,
        episode_id=episode_id,
        alert_id=scenario["id"],
        turn=turn,
        max_turns=max_turns,
        alert_text=turn_data["alert_text"],
        alert_source=turn_data["alert_source"],
        timestamp=turn_data.get("timestamp", ""),
        source_ip=turn_data.get("source_ip"),
        dest_ip=turn_data.get("dest_ip"),
        hostname=turn_data.get("hostname"),
        user_account=turn_data.get("user_account"),
        additional_context=turn_data.get("additional_context"),
        conversation_history=conversation_history or [],
        analyst_prompt=(
            turn_data.get("analyst_prompt")
            or (
                f"Turn {turn}/{max_turns}: Review the latest context and continue your investigation. "
                f"{'On this final turn, provide your complete MITRE classification and severity.' if turn == max_turns else ''}"
            )
        ),
        done=(turn == max_turns),
        reward=0.0,
    )


# ---------------------------------------------------------------------------
# Grader (evaluated once at episode end)
# ---------------------------------------------------------------------------

def grade(action: SOCAlertAction, scenario: dict) -> Tuple[float, Dict, str]:
    """
    Deterministic MITRE lookup grader for task2.

    The grade is based on the LAST action submitted (final turn).
    """
    score = 0.0
    breakdown: Dict[str, float] = {
        "mitre_tactic":           0.0,
        "severity":               0.0,
        "attack_started_at_turn": 0.0,
    }
    feedback_parts: list[str] = []

    ground_tactic  = scenario.get("ground_truth_mitre_tactic")
    ground_tactics = scenario.get("ground_truth_mitre_tactics", [])
    ground_sev     = scenario.get("ground_truth_severity")
    ground_turn    = scenario.get("attack_started_at_turn")

    # False-positive scenarios: no attack → correct answer is "no attack found"
    is_fp = scenario.get("attack_type") == "false_positive"

    # ── Dimension 1: MITRE Tactic (weight 0.40) ─────────────────────────
    if is_fp:
        # For FP scenarios, agent should NOT give a tactic
        if action.mitre_tactic is None or action.mitre_tactic == "" or action.mitre_tactic == "N/A":
            score += 0.40
            breakdown["mitre_tactic"] = 0.40
            feedback_parts.append("✅ Correctly identified no MITRE tactic — this is a false positive (+0.40)")
        else:
            feedback_parts.append(
                f"❌ Incorrectly assigned MITRE tactic '{action.mitre_tactic}' to a false positive (+0.0)"
            )
    else:
        agent_tactic = action.mitre_tactic or ""
        # Accept exact match OR accept if agent tactic is in the list of valid tactics
        if agent_tactic == ground_tactic or agent_tactic in ground_tactics:
            score += 0.40
            breakdown["mitre_tactic"] = 0.40
            feedback_parts.append(
                f"✅ Correct MITRE tactic '{agent_tactic}' (+0.40)"
            )
        else:
            feedback_parts.append(
                f"❌ Wrong MITRE tactic: got '{agent_tactic}', expected one of {ground_tactics} (+0.0)"
            )

    # ── Dimension 2: Severity (weight 0.30) ─────────────────────────────
    agent_sev = action.severity.value if action.severity else None
    if is_fp:
        if agent_sev in ("P4", None):  # FP should be P4 or no severity
            score += 0.30
            breakdown["severity"] = 0.30
            feedback_parts.append("✅ Correctly assigned P4 (low) severity to false positive (+0.30)")
        else:
            feedback_parts.append(
                f"❌ Over-triaged false positive as '{agent_sev}' — should be P4 (+0.0)"
            )
    else:
        if agent_sev == ground_sev:
            score += 0.30
            breakdown["severity"] = 0.30
            feedback_parts.append(f"✅ Correct severity '{ground_sev}' (+0.30)")
        else:
            # Give partial credit for being 1 tier off (e.g. P1 vs P2)
            if agent_sev and ground_sev:
                agent_num = int(agent_sev[1])
                ground_num = int(ground_sev[1])
                if abs(agent_num - ground_num) == 1:
                    score += 0.15
                    breakdown["severity"] = 0.15
                    feedback_parts.append(
                        f"⚠️ Close — severity '{agent_sev}' is one tier off from '{ground_sev}' (+0.15)"
                    )
                else:
                    feedback_parts.append(
                        f"❌ Wrong severity: got '{agent_sev}', expected '{ground_sev}' (+0.0)"
                    )
            else:
                feedback_parts.append(
                    f"❌ Severity not provided, expected '{ground_sev}' (+0.0)"
                )

    # ── Dimension 3: Attack started at turn (weight 0.30) ───────────────
    agent_turn  = action.attack_started_at_turn
    if is_fp:
        # FP: no attack, so turn should be None or 0
        if agent_turn is None or agent_turn == 0:
            score += 0.30
            breakdown["attack_started_at_turn"] = 0.30
            feedback_parts.append("✅ Correctly reported no attack turn for false positive (+0.30)")
        else:
            feedback_parts.append(
                f"❌ Incorrectly claimed attack started at turn {agent_turn} for a false positive (+0.0)"
            )
    else:
        if agent_turn == ground_turn:
            score += 0.30
            breakdown["attack_started_at_turn"] = 0.30
            feedback_parts.append(f"✅ Correct attack start turn {ground_turn} (+0.30)")
        else:
            feedback_parts.append(
                f"❌ Wrong attack start turn: got {agent_turn}, expected {ground_turn} (+0.0)"
            )

    return score, breakdown, "  ".join(feedback_parts)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def get_ground_truth(scenario: dict) -> dict:
    return {
        "attack_type":              scenario.get("attack_type"),
        "attack_started_at_turn":    scenario.get("attack_started_at_turn"),
        "attack_sophistication":     scenario.get("attack_sophistication"),
        "ground_truth_mitre_tactic":    scenario.get("ground_truth_mitre_tactic"),
        "ground_truth_mitre_technique":  scenario.get("ground_truth_mitre_technique"),
        "ground_truth_mitre_tactics":    scenario.get("ground_truth_mitre_tactics"),
        "ground_truth_severity":     scenario.get("ground_truth_severity"),
    }


# ---------------------------------------------------------------------------
# Task info
# ---------------------------------------------------------------------------

TASK_INFO = {
    "id":          "task2_investigation",
    "name":        "Multi-Turn MITRE ATT&CK Investigation",
    "difficulty":  "medium",
    "description": (
        "A multi-turn SOC investigation. The agent receives 3–4 turns of escalating "
        "context about an incident. On each turn, new forensic evidence is revealed. "
        "The agent must identify the MITRE ATT&CK tactic, severity, and which turn "
        "the attack began. Grader is a deterministic lookup table — MITRE ATT&CK "
        "framework is public documentation, not invented scoring."
    ),
    "scoring": {
        "mitre_tactic":           "0.40 — correct MITRE tactic ID (e.g. TA0001)",
        "severity":               "0.30 — correct severity P1–P4 (partial credit for ±1 tier)",
        "attack_started_at_turn": "0.30 — correct turn identification",
    },
    "grader_type": "deterministic",
    "max_turns":   "3–4 (varies by scenario)",
}
