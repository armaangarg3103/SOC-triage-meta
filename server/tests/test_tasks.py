"""
Tests for all 3 SOC Alert Triage task graders.

Run:
  pytest server/tests/ -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models import AlertType, SOCAlertAction, Severity
from server.tasks import task1_classification, task2_investigation, task3_response


# ---------------------------------------------------------------------------
# Task 1 — Classification grader
# ---------------------------------------------------------------------------

class TestTask1Classification:

    def setup_method(self):
        # Use a known real alert scenario
        self.real_scenario = {
            "id": "test_001",
            "alert_type": "phishing",
            "is_real_alert": True,
            "alert_source": "email_gateway",
            "timestamp": "2024-01-01T00:00:00Z",
            "alert_text": "Phishing email detected.",
            "ground_truth_mitre_tactic": "TA0001",
            "ground_truth_mitre_technique": "T1566",
            "ground_truth_severity": "P2",
            "ground_truth_containment": ["Block sender"],
        }
        self.fp_scenario = {
            "id": "test_002",
            "alert_type": "false_positive",
            "is_real_alert": False,
            "alert_source": "ids",
            "timestamp": "2024-01-01T00:00:00Z",
            "alert_text": "Authorized Nessus scan detected.",
            "ground_truth_mitre_tactic": None,
            "ground_truth_mitre_technique": None,
            "ground_truth_severity": "P4",
            "ground_truth_containment": ["No action required."],
        }

    def test_perfect_score_real_alert(self):
        action = SOCAlertAction(
            is_real_alert=True,
            alert_type=AlertType.phishing,
            confidence=1.0,
        )
        score, breakdown, feedback = task1_classification.grade(action, self.real_scenario)
        assert score == pytest.approx(1.0)
        assert breakdown["real_alert_detection"] == pytest.approx(0.5)
        assert breakdown["alert_type_classification"] == pytest.approx(0.5)

    def test_wrong_alert_type_loses_half(self):
        action = SOCAlertAction(
            is_real_alert=True,
            alert_type=AlertType.malware,
            confidence=0.9,
        )
        score, breakdown, _ = task1_classification.grade(action, self.real_scenario)
        assert score == pytest.approx(0.5)
        assert breakdown["alert_type_classification"] == pytest.approx(0.0)

    def test_false_positive_detected_correctly(self):
        action = SOCAlertAction(is_real_alert=False, confidence=1.0)
        score, breakdown, _ = task1_classification.grade(action, self.fp_scenario)
        assert score == pytest.approx(0.5)   # no alert_type required for FP

    def test_missed_false_positive_zero_score(self):
        action = SOCAlertAction(
            is_real_alert=True,
            alert_type=AlertType.malware,
            confidence=0.5,
        )
        score, _, _ = task1_classification.grade(action, self.fp_scenario)
        assert score == pytest.approx(0.0)

    def test_no_alert_type_provided_for_real_alert(self):
        action = SOCAlertAction(is_real_alert=True, confidence=0.9)
        score, breakdown, feedback = task1_classification.grade(action, self.real_scenario)
        assert score == pytest.approx(0.5)
        assert breakdown["alert_type_classification"] == pytest.approx(0.0)
        assert "not provided" in feedback

    def test_build_scenario_returns_dict(self):
        scenarios = task1_classification._load_alerts()
        assert len(scenarios) > 0
        s = task1_classification.build_scenario()
        assert "id" in s
        assert "alert_text" in s
        assert "is_real_alert" in s

    def test_build_observation(self):
        s   = task1_classification.build_scenario()
        obs = task1_classification.build_observation(s, "ep-test-001")
        assert obs.episode_id == "ep-test-001"
        assert obs.max_turns == 1
        assert obs.done is False


# ---------------------------------------------------------------------------
# Task 2 — Investigation grader
# ---------------------------------------------------------------------------

class TestTask2Investigation:

    def setup_method(self):
        self.scenario = {
            "id": "inv_test",
            "title": "Test Investigation",
            "attack_type": "malware",
            "attack_started_at_turn": 1,
            "attack_sophistication": "high",
            "ground_truth_mitre_tactic": "TA0011",
            "ground_truth_mitre_technique": "T1071",
            "ground_truth_mitre_tactics": ["TA0011", "TA0002"],
            "ground_truth_severity": "P1",
            "max_turns": 3,
            "turns": [{"turn": 1, "alert_text": "C2 beacon detected.", "alert_source": "edr"}],
        }
        self.fp_scenario = {
            "id": "inv_fp",
            "title": "FP Investigation",
            "attack_type": "false_positive",
            "attack_started_at_turn": None,
            "attack_sophistication": "low",
            "ground_truth_mitre_tactic": None,
            "ground_truth_mitre_technique": None,
            "ground_truth_mitre_tactics": [],
            "ground_truth_severity": "P4",
            "max_turns": 2,
            "turns": [{"turn": 1, "alert_text": "Authorized scan.", "alert_source": "ids"}],
        }

    def test_perfect_score(self):
        action = SOCAlertAction(
            mitre_tactic="TA0011",
            mitre_technique="T1071",
            severity=Severity.P1,
            attack_started_at_turn=1,
        )
        score, breakdown, _ = task2_investigation.grade(action, self.scenario)
        assert score == pytest.approx(1.0)

    def test_accepts_alternative_valid_tactic(self):
        action = SOCAlertAction(
            mitre_tactic="TA0002",  # also in ground_truth_mitre_tactics
            severity=Severity.P1,
            attack_started_at_turn=1,
        )
        score, breakdown, _ = task2_investigation.grade(action, self.scenario)
        assert breakdown["mitre_tactic"] == pytest.approx(0.40)

    def test_partial_credit_severity_one_tier_off(self):
        action = SOCAlertAction(
            mitre_tactic="TA0011",
            severity=Severity.P2,  # off by 1
            attack_started_at_turn=1,
        )
        score, breakdown, _ = task2_investigation.grade(action, self.scenario)
        assert breakdown["severity"] == pytest.approx(0.15)

    def test_fp_correct_no_tactic(self):
        action = SOCAlertAction(
            mitre_tactic=None,
            severity=Severity.P4,
            attack_started_at_turn=None,
        )
        score, _, _ = task2_investigation.grade(action, self.fp_scenario)
        assert score == pytest.approx(1.0)

    def test_fp_wrong_tactic_loses_points(self):
        action = SOCAlertAction(
            mitre_tactic="TA0011",
            severity=Severity.P1,
            attack_started_at_turn=1,
        )
        score, _, _ = task2_investigation.grade(action, self.fp_scenario)
        assert score == pytest.approx(0.0)

    def test_load_investigations(self):
        invs = task2_investigation._load_investigations()
        assert len(invs) > 0
        for inv in invs:
            assert "id" in inv
            assert "turns" in inv
            assert len(inv["turns"]) == inv["max_turns"]


# ---------------------------------------------------------------------------
# Task 3 — Response grader
# ---------------------------------------------------------------------------

class TestTask3Response:

    def setup_method(self):
        self.scenario = {
            "id": "alert_test",
            "alert_type": "malware",
            "is_real_alert": True,
            "alert_source": "edr",
            "timestamp": "2024-01-01T00:00:00Z",
            "alert_text": "Cobalt Strike beacon detected via reflective DLL injection.",
            "ground_truth_mitre_tactic": "TA0011",
            "ground_truth_mitre_technique": "T1071",
            "ground_truth_severity": "P1",
            "ground_truth_containment": [
                "Isolate host from network",
                "Kill malicious process and dump memory",
                "Block C2 IP at firewall",
                "Run threat hunt",
                "Engage IR team",
            ],
            "affected_systems": ["WKSTN-001"],
        }

    def test_mitre_in_reasoning_scores(self):
        action = SOCAlertAction(
            incident_summary="A Cobalt Strike beacon using T1071 was detected.",
            containment_steps=["Isolate host","Block C2 IP","Kill process","Engage IR team","Dump memory"],
            affected_systems=["WKSTN-001"],
            escalate_to_ir=True,
            mitre_technique="T1071",
            reasoning="T1071 C2 over HTTP used by Cobalt Strike.",
        )
        score, breakdown, _ = task3_response.grade(action, self.scenario)
        assert breakdown["mitre_technique_accuracy"] == pytest.approx(0.30)
        assert breakdown["escalation_decision"] == pytest.approx(0.20)
        assert score >= 0.50   # MITRE + escalation + some containment credit

    def test_missing_mitre_loses_points(self):
        action = SOCAlertAction(
            incident_summary="A malware incident occurred.",
            containment_steps=["Isolate the host"],
            escalate_to_ir=True,
        )
        score, breakdown, _ = task3_response.grade(action, self.scenario)
        assert breakdown["mitre_technique_accuracy"] == pytest.approx(0.0)

    def test_wrong_escalation_loses_points(self):
        action = SOCAlertAction(
            incident_summary="T1071 T1071 T1071 beacon detected containment isolate block",
            containment_steps=["Isolate", "Block", "Hunt", "IR", "Memory"],
            mitre_technique="T1071",
            escalate_to_ir=False,  # WRONG — P1 should escalate
        )
        score, breakdown, _ = task3_response.grade(action, self.scenario)
        assert breakdown["escalation_decision"] == pytest.approx(0.0)

    def test_heuristic_fallback_works(self):
        """Heuristic judge should not crash and should return 0.0-1.0."""
        action = SOCAlertAction(
            incident_summary="Security incident detected. Network isolation required immediately.",
            containment_steps=["Isolate host","Block IP","Reset credentials","Engage IR","Forensic image"],
            escalate_to_ir=True,
        )
        quality = task3_response._heuristic_judge(self.scenario, action)
        assert 0.0 <= quality <= 1.0

    def test_no_steps_zero_containment(self):
        action = SOCAlertAction(
            incident_summary="Incident occurred.",
            containment_steps=[],
            escalate_to_ir=True,
        )
        score, breakdown, feedback = task3_response.grade(action, self.scenario)
        assert breakdown["containment_quality"] == pytest.approx(0.0)
        assert "No containment steps" in feedback


# ---------------------------------------------------------------------------
# Integration — full episode via environment class
# ---------------------------------------------------------------------------

class TestEnvironmentIntegration:

    def test_task1_full_episode(self):
        from server.environment import SOCAlertEnvironment
        env = SOCAlertEnvironment()
        obs = env.reset(task_id="task1_classification")
        assert obs.task_id.value == "task1_classification"
        assert obs.episode_id is not None
        assert obs.done is False

        action = SOCAlertAction(is_real_alert=True, alert_type=AlertType.malware)
        obs2 = env.step(action)
        assert obs2.done is True

        result = env.grade_episode()
        assert result.episode_id == obs.episode_id
        assert 0.0 <= result.final_score <= 1.0
        assert "final_score" in result.model_dump()

    def test_task2_full_episode(self):
        from server.environment import SOCAlertEnvironment
        env = SOCAlertEnvironment()
        obs = env.reset(task_id="task2_investigation")
        assert obs.task_id.value == "task2_investigation"
        assert obs.max_turns >= 2

        action = SOCAlertAction(
            mitre_tactic="TA0011",
            severity=Severity.P1,
            attack_started_at_turn=1,
        )
        # Step through all turns
        current_obs = obs
        while not current_obs.done:
            current_obs = env.step(action)

        result = env.grade_episode()
        assert 0.0 <= result.final_score <= 1.0

    def test_task3_full_episode(self):
        from server.environment import SOCAlertEnvironment
        env = SOCAlertEnvironment()
        obs = env.reset(task_id="task3_response")
        assert obs.task_id.value == "task3_response"

        action = SOCAlertAction(
            incident_summary="Major incident. T1566 phishing attack. Isolate and remediate.",
            containment_steps=["Isolate","Block","Reset password","Enable MFA","Engage IR"],
            escalate_to_ir=True,
            mitre_technique="T1566",
        )
        obs2 = env.step(action)
        assert obs2.done is True
        result = env.grade_episode()
        assert 0.0 <= result.final_score <= 1.0

    def test_reset_without_step_still_grades(self):
        from server.environment import SOCAlertEnvironment
        env = SOCAlertEnvironment()
        env.reset(task_id="task1_classification")
        result = env.grade_episode()
        assert result.final_score == 0.0
