"""
SOC Alert Triage Environment — Shared Data Models

This module defines every schema used across:
  server/app.py, server/environment.py, all task modules,
  baseline.py, inference.py

If you change anything here, keep ALL consumers in sync.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TaskID(str, Enum):
    task1_classification = "task1_classification"
    task2_investigation  = "task2_investigation"
    task3_response       = "task3_response"


class AlertType(str, Enum):
    phishing          = "phishing"
    malware           = "malware"
    lateral_movement  = "lateral_movement"
    data_exfil        = "data_exfil"
    false_positive    = "false_positive"


class Severity(str, Enum):
    P1 = "P1"   # Critical  — immediate response
    P2 = "P2"   # High      — respond within 1 hour
    P3 = "P3"   # Medium    — respond within 4 hours
    P4 = "P4"   # Low       — respond within 24 hours


class AlertSource(str, Enum):
    siem          = "siem"
    ids           = "ids"
    edr           = "edr"
    email_gateway = "email_gateway"
    dlp           = "dlp"
    firewall      = "firewall"
    threat_intel  = "threat_intel"


# ---------------------------------------------------------------------------
# Observation — what the agent receives
# ---------------------------------------------------------------------------

class SOCAlertObservation(BaseModel):
    """Observation returned to the agent after reset or step."""

    task_id:    TaskID
    episode_id: str
    alert_id:   str

    # Turn tracking (task2 uses multi-turn)
    turn:      int = 1
    max_turns: int = 1

    # Core alert fields
    alert_text:   str
    alert_source: str
    timestamp:    str

    # Network / host context (optional — may be null for some alerts)
    source_ip:    Optional[str] = None
    dest_ip:      Optional[str] = None
    hostname:     Optional[str] = None
    user_account: Optional[str] = None
    raw_log:      Optional[str] = None

    # Multi-turn investigation state
    conversation_history: List[Dict] = Field(default_factory=list)
    additional_context:   Optional[str] = None
    analyst_prompt:       Optional[str] = None   # Prompt shown to agent each turn

    # Episode state (populated after step / at done=True)
    done:            bool  = False
    reward:          float = 0.0
    score_breakdown: Dict  = Field(default_factory=dict)
    feedback:        str   = ""


# ---------------------------------------------------------------------------
# Action — what the agent submits
# ---------------------------------------------------------------------------

class SOCAlertAction(BaseModel):
    """Action submitted by the agent via POST /step_task."""

    # ── Task 1: Alert Classification ──────────────────────────────────────
    alert_type:    Optional[AlertType] = None
    is_real_alert: Optional[bool]      = None   # False = false positive

    # ── Task 2: Deep Investigation ────────────────────────────────────────
    mitre_tactic:          Optional[str]      = None   # e.g. "TA0001"
    mitre_technique:       Optional[str]      = None   # e.g. "T1566"
    severity:              Optional[Severity] = None
    attack_started_at_turn: Optional[int]    = None

    # ── Task 3: Incident Response ─────────────────────────────────────────
    incident_summary:  Optional[str]       = None
    containment_steps: Optional[List[str]] = None
    affected_systems:  Optional[List[str]] = None
    escalate_to_ir:    Optional[bool]      = None

    # ── Common ─────────────────────────────────────────────────────────────
    confidence: float          = Field(default=1.0, ge=0.0, le=1.0)
    reasoning:  Optional[str]  = None


# ---------------------------------------------------------------------------
# Episode result — returned by /grader
# ---------------------------------------------------------------------------

class EpisodeResult(BaseModel):
    episode_id:      str
    task_id:         str
    final_score:     float
    score_breakdown: Dict
    feedback:        str
    ground_truth:    Dict
    agent_actions:   List[Dict] = Field(default_factory=list)
