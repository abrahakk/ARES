from __future__ import annotations

import logging
from typing import List, Optional

from coordination.state_manager import StateManager
from missions.mission import Mission
from rl.coordination.formation import FormationController

logger = logging.getLogger(__name__)


class FailureRecoveryManager:
    def __init__(self, mission: Mission, state_manager: StateManager) -> None:
        self.mission = mission
        self.state_manager = state_manager
        self.formation = FormationController()

    async def check_for_silent_failures(self, staleness_threshold_s: float = 5.0) -> List[str]:
        newly_failed: List[str] = []
        for robot_id in self.mission.robot_ids:
            state = await self.state_manager.get_state(robot_id)
            if state is None or state.is_stale(staleness_threshold_s):
                logger.warning("robot=%s appears failed (stale or missing state)", robot_id)
                await self.mission.report_robot_failure(robot_id)
                newly_failed.append(robot_id)
        return newly_failed

    async def suggest_regroup_point(self, exclude_robot_id: Optional[str] = None):
        
        remaining_ids = [rid for rid in self.mission.robot_ids if rid != exclude_robot_id]
        states = await self.state_manager.get_all_states(remaining_ids)
        if not states:
            return None

        positions = {robot_id: state.pose.as_tuple()[:2] for robot_id, state in states.items()}
        rendezvous = self.formation.compute_rendezvous_point(positions)
        logger.info("suggested regroup point after failure: %s", rendezvous)
        return rendezvous
