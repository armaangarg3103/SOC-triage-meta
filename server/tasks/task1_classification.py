"""
Task 1 — Alert Classification (Easy)

Agent receives a single SOC alert and must:
  1. Decide if it is a real threat (is_real_alert: bool)
  2. Classify the alert type (alert_type: AlertType enum)

Grader: Purely deterministic lookup — no LLM required.
  +0.5 for correct is_real_alert
  +0.5 for correct alert_type  (only evaluated when is_real_alert=True)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

from models import AlertType, SOCAlertAction, SOCAlertObservation, TaskID

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_DATA_FILE = Path(__file__).parent.parent / "data" / "alerts.json"
_ALERTS: list[dict] = []


def _load_alerts() -> list[dict]:
    global _ALERTS
    if not _ALERTS:
        with open(_DATA_FILE, encoding="utf-8") as f:
            all_alerts = json.load(f)
        # task1 uses alerts flagged "task1" or "both"
        _ALERTS = [a for a in all_alerts if a.get("task") in ("task1", "both")]
    return _ALERTS


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------

def build_scenario(alert_id: Optional[str] = None) -> dict:
    """Return a random (or specific) alert scenario dict."""
    alerts = _load_alerts()
    if alert_id:
        scenario = next((a for a in alerts if a["id"] == alert_id), None)
        if scenario is None:
            raise ValueError(f"Alert ID '{alert_id}' not found in task1 data")
        return scenario
    return random.choice(alerts)


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

def build_observation(scenario: dict, episode_id: str) -> SOCAlertObservation:
    """Convert a scenario dict into the observation the agent receives."""
    return SOCAlertObservation(
        task_id=TaskID.task1_classification,
        episode_id=episode_id,
        alert_id=scenario["id"],
        turn=1,
        max_turns=1,
        alert_text=scenario["alert_text"],
        alert_source=scenario["alert_source"],
        timestamp=scenario["timestamp"],
        source_ip=scenario.get("source_ip"),
        dest_ip=scenario.get("dest_ip"),
        hostname=scenario.get("hostname"),
        user_account=scenario.get("user_account"),
        raw_log=scenario.get("raw_log"),
        analyst_prompt=(
            "You are a Tier-2 SOC analyst. Review the alert above and:\n"
            "1. Determine if this is a real security threat or a false positive.\n"
            "2. If it is a real threat, classify the alert type.\n"
            "Respond using the structured action format."
        ),
        done=False,
        reward=0.0,
    )


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def grade(action: SOCAlertAction, scenario: dict) -> Tuple[float, Dict, str]:
    """
    Deterministic grader for task1.

    Returns:
        score         (0.0–1.0)
        breakdown     dict with per-dimension scores
        feedback      human-readable explanation
    """
    score = 0.0
    breakdown: Dict[str, float] = {
        "real_alert_detection": 0.0,
        "alert_type_classification": 0.0,
    }
    feedback_parts: list[str] = []

    ground_is_real = scenario["is_real_alert"]
    ground_type    = scenario.get("alert_type")   # may be null for confirmed FPs

    # ── Dimension 1: Correct real/false-positive classification ─────────
    agent_is_real = action.is_real_alert
    if agent_is_real is None:
        feedback_parts.append("❌ is_real_alert was not provided.")
    elif agent_is_real == ground_is_real:
        score += 0.5
        breakdown["real_alert_detection"] = 0.5
        feedback_parts.append("✅ Correctly identified threat/false-positive status (+0.5)")
    else:
        label = "real threat" if ground_is_real else "false positive"
        feedback_parts.append(
            f"❌ Missed — this alert is a {label}. is_real_alert should be {ground_is_real}. (+0.0)"
        )

    # ── Dimension 2: Correct alert type (only for real alerts) ──────────
    if ground_is_real:
        agent_type = action.alert_type
        if agent_type is None:
            feedback_parts.append("❌ alert_type was not provided for a real threat. (+0.0)")
        elif agent_type.value == ground_type:
            score += 0.5
            breakdown["alert_type_classification"] = 0.5
            feedback_parts.append(
                f"✅ Correctly classified as '{ground_type}' (+0.5)"
            )
        else:
            feedback_parts.append(
                f"❌ Wrong alert type: got '{agent_type.value}', expected '{ground_type}'. (+0.0)"
            )
    else:
        feedback_parts.append(
            f"ℹ️ Alert type scoring skipped — this is a false positive."
        )

    return score, breakdown, "  ".join(feedback_parts)


# ---------------------------------------------------------------------------
# Ground truth (for /grader endpoint)
# ---------------------------------------------------------------------------

def get_ground_truth(scenario: dict) -> dict:
    return {
        "is_real_alert":   scenario["is_real_alert"],
        "alert_type":      scenario.get("alert_type"),
        "severity":        scenario.get("ground_truth_severity"),
        "mitre_tactic":    scenario.get("ground_truth_mitre_tactic"),
        "mitre_technique": scenario.get("ground_truth_mitre_technique"),
    }


# ---------------------------------------------------------------------------
# Task info (shown at /tasks endpoint)
# ---------------------------------------------------------------------------

TASK_INFO = {
    "id":          "task1_classification",
    "name":        "SOC Alert Classification",
    "difficulty":  "easy",
    "description": (
        "The agent receives a single SOC alert from SIEM/IDS/EDR/email gateway. "
        "It must (1) determine if the alert is a real threat or a false positive, "
        "and (2) classify the alert type (phishing, malware, lateral_movement, "
        "data_exfil, or false_positive)."
    ),
    "scoring": {
        "real_alert_detection":    "0.5 — correct is_real_alert detection",
        "alert_type_classification": "0.5 — correct alert_type classification (real alerts only)",
    },
    "grader_type": "deterministic",
    "max_turns":   1,
    "total_scenarios": None,   # filled at runtime
}
