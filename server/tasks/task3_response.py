"""
Task 3 — Incident Response (Hard)

Agent receives a high-severity SOC alert and must produce:
  - incident_summary   : executive-quality incident summary
  - containment_steps  : ordered list of containment actions
  - affected_systems   : list of impacted systems
  - escalate_to_ir     : bool — should IR team be engaged?
  - mitre_technique    : MITRE technique ID for the attack

Grader: Hybrid — deterministic MITRE check + heuristic scoring
        + optional LLM judge (Groq/OpenAI-compatible).

  +0.30 — MITRE technique referenced in summary/reasoning
  +0.30 — Sufficient containment steps (≥3 steps, quality check)
  +0.20 — Correct escalation decision
  +0.20 — LLM judge OR heuristic quality score on incident_summary
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from models import AlertType, SOCAlertAction, SOCAlertObservation, TaskID

# ---------------------------------------------------------------------------
# Data loading — reuses the alerts.json data (task3 uses "both" tagged alerts)
# ---------------------------------------------------------------------------

_ALERTS_FILE = Path(__file__).parent.parent / "data" / "alerts.json"
_ALERTS: list[dict] = []


def _load_alerts() -> list[dict]:
    global _ALERTS
    if not _ALERTS:
        with open(_ALERTS_FILE, encoding="utf-8") as f:
            all_alerts = json.load(f)
        _ALERTS = [a for a in all_alerts if a.get("task") in ("task3", "both") and a["is_real_alert"]]
    return _ALERTS


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------

def build_scenario(alert_id: Optional[str] = None) -> dict:
    alerts = _load_alerts()
    if alert_id:
        scenario = next((a for a in alerts if a["id"] == alert_id), None)
        if scenario is None:
            raise ValueError(f"Alert ID '{alert_id}' not found in task3 data")
        return scenario
    return random.choice(alerts)


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

def build_observation(scenario: dict, episode_id: str) -> SOCAlertObservation:
    return SOCAlertObservation(
        task_id=TaskID.task3_response,
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
            "You are an experienced Incident Responder. This is a confirmed security incident. "
            "Provide:\n"
            "1. A clear incident_summary (2–4 sentences, executive-ready)\n"
            "2. An ordered list of containment_steps (specific, actionable)\n"
            "3. A list of affected_systems\n"
            "4. Whether to escalate_to_ir (true/false)\n"
            "5. The MITRE ATT&CK technique (mitre_technique) for this attack\n"
            "6. Your reasoning"
        ),
        done=False,
        reward=0.0,
    )


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

def _call_llm_judge(
    scenario: dict,
    action: SOCAlertAction,
) -> float:
    """
    Call an LLM (via OpenAI) to score the incident response.
    Returns a float 0.0–1.0. Falls back to heuristic on any exception.
    """
    try:
        from openai import OpenAI
        
        api_key  = os.getenv("OPENAI_API_KEY", "")
        api_base = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
        
        if not api_key:
            return _heuristic_judge(scenario, action)
        
        client = OpenAI(api_key=api_key, base_url=api_base)
        model  = os.getenv("MODEL_NAME", "gpt-4o-mini")

        judge_prompt = f"""You are an expert SOC analyst and incident response trainer.

ORIGINAL ALERT:
{scenario['alert_text']}

GROUND TRUTH CONTAINMENT STEPS:
{json.dumps(scenario.get('ground_truth_containment', []), indent=2)}

AGENT'S INCIDENT SUMMARY:
{action.incident_summary or "(not provided)"}

AGENT'S CONTAINMENT STEPS:
{json.dumps(action.containment_steps or [], indent=2)}

AGENT'S REASONING:
{action.reasoning or "(not provided)"}

Score the agent's response on a scale of 0.0 to 1.0 based on:
- Does the incident summary accurately describe the threat?
- Are the containment steps appropriate, specific, and in the right priority order?
- Are critical actions (isolation, credential reset, evidence preservation) included?
- Is the response proportionate to the severity of the incident?

Respond with ONLY a JSON object: {{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}"""

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=150,
        )

        content = response.choices[0].message.content.strip()
        # Extract JSON from response
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            result = json.loads(match.group())
            raw_score = float(result.get("score", 0.5))
            return max(0.0, min(1.0, raw_score))
        return _heuristic_judge(scenario, action)

    except Exception:
        return _heuristic_judge(scenario, action)


def _heuristic_judge(scenario: dict, action: SOCAlertAction) -> float:
    """
    Fallback heuristic judge that doesn't require an LLM.
    Checks for key response quality signals.
    """
    score = 0.0
    summary = (action.incident_summary or "").lower()
    steps   = action.containment_steps or []
    reasoning = (action.reasoning or "").lower()

    # Signal 1: Summary mentions key terms related to the alert (0.25)
    alert_keywords = _extract_keywords(scenario["alert_text"])
    mentioned = sum(1 for kw in alert_keywords if kw in summary or kw in reasoning)
    score += min(0.25, (mentioned / max(len(alert_keywords), 1)) * 0.25)

    # Signal 2: At least 3 containment steps (0.25)
    if len(steps) >= 5:
        score += 0.25
    elif len(steps) >= 3:
        score += 0.15
    elif len(steps) >= 1:
        score += 0.05

    # Signal 3: Critical action keywords present in steps (0.25)
    critical_terms = ["isolat", "block", "disable", "reset", "revoke", "quarantin", "preserv", "forensic"]
    steps_text = " ".join(steps).lower()
    critical_hits = sum(1 for t in critical_terms if t in steps_text)
    score += min(0.25, (critical_hits / len(critical_terms)) * 0.25)

    # Signal 4: Summary is substantive (not empty/placeholder) (0.25)
    if len(summary) > 150:
        score += 0.25
    elif len(summary) > 75:
        score += 0.10

    return min(1.0, score)


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful lowercase keyword tokens from alert text."""
    # Remove punctuation, lowercase, split
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    stopwords = {"this", "that", "with", "from", "have", "been", "were", "will", "they",
                 "their", "also", "over", "into", "when", "than", "then", "some"}
    return [w for w in words if w not in stopwords][:15]


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def grade(action: SOCAlertAction, scenario: dict) -> Tuple[float, Dict, str]:
    """
    Hybrid grader for task3:
      +0.30 MITRE technique in summary/reasoning
      +0.30 Sufficient + quality containment steps
      +0.20 Correct escalation decision
      +0.20 LLM judge / heuristic for incident summary quality
    """
    score = 0.0
    breakdown: Dict[str, float] = {
        "mitre_technique_accuracy": 0.0,
        "containment_quality":      0.0,
        "escalation_decision":      0.0,
        "response_quality":         0.0,
    }
    feedback_parts: list[str] = []

    ground_technique = scenario.get("ground_truth_mitre_technique", "")
    ground_sev       = scenario.get("ground_truth_severity", "")
    ground_steps     = scenario.get("ground_truth_containment", [])

    # High-severity alerts should be escalated to IR
    should_escalate = ground_sev in ("P1", "P2")

    # ── Dimension 1: MITRE technique (0.30) ─────────────────────────────
    combined_text = " ".join(filter(None, [
        action.incident_summary,
        action.reasoning,
        action.mitre_technique,
    ])).upper()

    if ground_technique and ground_technique.upper() in combined_text:
        score += 0.30
        breakdown["mitre_technique_accuracy"] = 0.30
        feedback_parts.append(f"✅ MITRE technique {ground_technique} correctly referenced (+0.30)")
    elif action.mitre_technique:
        # Partial credit if a related technique is provided
        ground_family = ground_technique.split(".")[0] if ground_technique else ""
        agent_family  = (action.mitre_technique or "").split(".")[0].upper()
        if ground_family and ground_family.upper() == agent_family:
            score += 0.15
            breakdown["mitre_technique_accuracy"] = 0.15
            feedback_parts.append(
                f"⚠️ Close — {action.mitre_technique} is the right technique family but wrong sub-technique. Expected {ground_technique} (+0.15)"
            )
        else:
            feedback_parts.append(
                f"❌ MITRE technique mismatch: got '{action.mitre_technique}', expected '{ground_technique}' (+0.0)"
            )
    else:
        feedback_parts.append(
            f"❌ MITRE technique not referenced. Expected {ground_technique} in summary or mitre_technique field (+0.0)"
        )

    # ── Dimension 2: Containment steps quality (0.30) ───────────────────
    steps = action.containment_steps or []
    if len(steps) == 0:
        feedback_parts.append("❌ No containment steps provided (+0.0)")
    else:
        # Check overlap with ground truth steps
        steps_text = " ".join(steps).lower()
        ground_keywords = []
        for gs in ground_steps:
            ground_keywords.extend(_extract_keywords(gs))
        ground_keywords = list(set(ground_keywords))

        matched = sum(1 for kw in ground_keywords if kw in steps_text)
        coverage = matched / max(len(ground_keywords), 1)

        step_score = min(0.30, coverage * 0.30)
        # Bonus for step count
        if len(steps) >= 5:
            step_score = min(0.30, step_score * 1.2)

        score += step_score
        breakdown["containment_quality"] = step_score
        feedback_parts.append(
            f"{'✅' if step_score >= 0.20 else '⚠️'} Containment coverage {coverage:.0%} across {len(steps)} steps (+{step_score:.2f})"
        )

    # ── Dimension 3: Escalation decision (0.20) ─────────────────────────
    agent_escalate = action.escalate_to_ir
    if agent_escalate is None:
        feedback_parts.append("❌ escalate_to_ir not provided (+0.0)")
    elif agent_escalate == should_escalate:
        score += 0.20
        breakdown["escalation_decision"] = 0.20
        label = "correctly escalated to IR" if should_escalate else "correctly not escalated (low severity)"
        feedback_parts.append(f"✅ {label} (+0.20)")
    else:
        if should_escalate:
            feedback_parts.append("❌ Should have escalated to IR for this severity (+0.0)")
        else:
            feedback_parts.append("❌ Unnecessarily escalated to IR for a low-severity alert (+0.0)")

    # ── Dimension 4: Response quality via LLM or heuristic (0.20) ───────
    quality_score = _call_llm_judge(scenario, action) * 0.20
    score += quality_score
    breakdown["response_quality"] = quality_score
    feedback_parts.append(f"ℹ️ Response quality score: {quality_score:.2f}/0.20")

    return score, breakdown, "  ".join(feedback_parts)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def get_ground_truth(scenario: dict) -> dict:
    return {
        "alert_type":               scenario["alert_type"],
        "is_real_alert":            scenario["is_real_alert"],
        "ground_truth_mitre_tactic":    scenario.get("ground_truth_mitre_tactic"),
        "ground_truth_mitre_technique":  scenario.get("ground_truth_mitre_technique"),
        "ground_truth_severity":     scenario.get("ground_truth_severity"),
        "ground_truth_containment":  scenario.get("ground_truth_containment", []),
        "affected_systems":          scenario.get("affected_systems", []),
    }


# ---------------------------------------------------------------------------
# Task info
# ---------------------------------------------------------------------------

TASK_INFO = {
    "id":          "task3_response",
    "name":        "Incident Response Summary",
    "difficulty":  "hard",
    "description": (
        "The agent receives a confirmed high-severity SOC alert and must produce "
        "a complete incident response: executive summary, ordered containment steps, "
        "affected system list, escalation decision, and MITRE ATT&CK technique. "
        "Scored by MITRE accuracy check + containment quality + LLM judge with heuristic fallback."
    ),
    "scoring": {
        "mitre_technique_accuracy": "0.30 — MITRE technique correctly referenced",
        "containment_quality":      "0.30 — Containment steps coverage and quality",
        "escalation_decision":      "0.20 — Correct IR escalation decision",
        "response_quality":         "0.20 — LLM judge or heuristic on incident_summary",
    },
    "grader_type": "hybrid (deterministic + LLM judge with fallback)",
    "max_turns":   1,
}
