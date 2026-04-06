"""
Main Environment Class — SOC Alert Triage

This is the single source of truth for episode logic.
server/__init__.py and server/app.py both import from here.

Episode flow:
  1. POST /reset_task   → environment.reset(task_id) → SOCAlertObservation
  2. POST /step_task    → environment.step(action)   → SOCAlertObservation (with reward)
  3. POST /grader       → environment.grade_episode() → EpisodeResult
"""

import uuid
from typing import Dict, List, Optional

from models import EpisodeResult, SOCAlertAction, SOCAlertObservation, TaskID
from server.tasks import task1_classification, task2_investigation, task3_response


class SOCAlertEnvironment:
    """
    Stateful environment for one episode at a time.

    In production (HF Space), each request is handled by a
    separate environment instance keyed by episode_id.
    """

    def __init__(self) -> None:
        self.episode_id:       Optional[str]     = None
        self.task_id:          Optional[TaskID]  = None
        self.scenario:         Optional[dict]    = None
        self.current_turn:     int               = 0
        self.max_turns:        int               = 1
        self.actions_history:  List[Dict]        = []
        self.final_score:      float             = 0.0
        self.score_breakdown:  Dict              = {}
        self.feedback:         str               = ""
        self.conversation_history: List[Dict]   = []
        self._last_observation: Optional[SOCAlertObservation] = None

    # -----------------------------------------------------------------------
    # reset
    # -----------------------------------------------------------------------

    def reset(
        self,
        task_id: str,
        episode_id: Optional[str] = None,
        alert_id: Optional[str] = None,
    ) -> SOCAlertObservation:
        """Start a new episode. Returns the first observation."""
        self.episode_id         = episode_id or str(uuid.uuid4())
        self.task_id            = TaskID(task_id)
        self.current_turn       = 1
        self.actions_history    = []
        self.conversation_history = []
        self.final_score        = 0.0
        self.score_breakdown    = {}
        self.feedback           = ""

        if self.task_id == TaskID.task1_classification:
            self.scenario = task1_classification.build_scenario(alert_id)
            self.max_turns = 1
            obs = task1_classification.build_observation(self.scenario, self.episode_id)

        elif self.task_id == TaskID.task2_investigation:
            self.scenario = task2_investigation.build_scenario(alert_id)
            self.max_turns = self.scenario["max_turns"]
            obs = task2_investigation.build_observation(
                self.scenario, self.episode_id, turn=1
            )

        elif self.task_id == TaskID.task3_response:
            self.scenario = task3_response.build_scenario(alert_id)
            self.max_turns = 1
            obs = task3_response.build_observation(self.scenario, self.episode_id)

        else:
            raise ValueError(f"Unknown task_id: {task_id!r}")

        self._last_observation = obs
        return obs

    # -----------------------------------------------------------------------
    # step
    # -----------------------------------------------------------------------

    def step(self, action: SOCAlertAction) -> SOCAlertObservation:
        """
        Process one agent action. Returns the next observation.
        On the final turn, populates reward, score_breakdown, feedback, done=True.
        """
        if self.scenario is None:
            raise RuntimeError("Call reset() before step()")

        # Record action
        self.actions_history.append(action.model_dump())

        is_final_turn = (self.current_turn >= self.max_turns)

        if is_final_turn:
            # Grade the episode
            score, breakdown, feedback = self._run_grader(action)
            self.final_score     = score
            self.score_breakdown = breakdown
            self.feedback        = feedback

            # Build terminal observation
            obs = self._last_observation.model_copy(update={
                "done":            True,
                "reward":          score,
                "score_breakdown": breakdown,
                "feedback":        feedback,
                "turn":            self.current_turn,
            })

        else:
            # Multi-turn: advance and return next turn's observation (task2 only)
            self.conversation_history.append({
                "turn":   self.current_turn,
                "action": action.model_dump(exclude_none=True),
            })
            self.current_turn += 1

            obs = task2_investigation.build_observation(
                self.scenario,
                self.episode_id,
                turn=self.current_turn,
                conversation_history=self.conversation_history,
            )

        self._last_observation = obs
        return obs

    # -----------------------------------------------------------------------
    # grade_episode
    # -----------------------------------------------------------------------

    def grade_episode(self) -> EpisodeResult:
        """Return the full episode result (called by POST /grader)."""
        if self.scenario is None:
            raise RuntimeError("No active episode — call reset() first")

        ground_truth = self._get_ground_truth()

        return EpisodeResult(
            episode_id=self.episode_id or "",
            task_id=self.task_id.value if self.task_id else "",
            final_score=self.final_score,
            score_breakdown=self.score_breakdown,
            feedback=self.feedback,
            ground_truth=ground_truth,
            agent_actions=self.actions_history,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _run_grader(self, action: SOCAlertAction):
        if self.task_id == TaskID.task1_classification:
            return task1_classification.grade(action, self.scenario)
        elif self.task_id == TaskID.task2_investigation:
            return task2_investigation.grade(action, self.scenario)
        elif self.task_id == TaskID.task3_response:
            return task3_response.grade(action, self.scenario)
        raise ValueError(f"No grader for task_id: {self.task_id}")

    def _get_ground_truth(self) -> dict:
        if self.task_id == TaskID.task1_classification:
            return task1_classification.get_ground_truth(self.scenario)
        elif self.task_id == TaskID.task2_investigation:
            return task2_investigation.get_ground_truth(self.scenario)
        elif self.task_id == TaskID.task3_response:
            return task3_response.get_ground_truth(self.scenario)
        return {}
