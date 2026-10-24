from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from coordination.state_manager import StateManager
from coordination.task_manager import TaskManager
from rl.coordination.task_allocation import AuctionTaskAllocator

logger = logging.getLogger(__name__)

Point = Tuple[float, float]

FRONTIER_REACHED_THRESHOLD_M = 0.5
FRONTIER_DISCOVERY_MIN_DISTANCE_M = 1.0  


class ExplorationManager:
    def __init__(self, task_manager: TaskManager, state_manager: StateManager) -> None:
        self.task_manager = task_manager
        self.state_manager = state_manager
        self.allocator = AuctionTaskAllocator()

    async def register_discovered_frontier(self, position: Point, discovered_by: str) -> str:
        
        existing_tasks = await self.task_manager.list_active_tasks()
        for task in existing_tasks:
            existing_position = task.get("position")
            if existing_position is None:
                continue
            distance = ((existing_position[0] - position[0]) ** 2 + (existing_position[1] - position[1]) ** 2) ** 0.5
            if distance < FRONTIER_DISCOVERY_MIN_DISTANCE_M:
                return task["frontier_id"]

        frontier_id = f"discovered_{discovered_by}_{len(existing_tasks):04d}"
        await self.task_manager.register_frontier(frontier_id, position)
        logger.debug("robot=%s discovered new frontier=%s at %s", discovered_by, frontier_id, position)
        return frontier_id

    async def allocate_idle_robots(self, robot_ids: List[str]) -> Dict[str, Point]:
        
        active_tasks = await self.task_manager.list_active_tasks()
        unclaimed = [tuple(t["position"]) for t in active_tasks if not t.get("claimed_by")]
        if not unclaimed:
            return {}

        robot_states = await self.state_manager.get_all_states(robot_ids)
        idle_positions = {
            robot_id: state.pose.as_tuple()[:2]
            for robot_id, state in robot_states.items()
            if state.current_goal is None
        }
        if not idle_positions:
            return {}

        proposed = self.allocator.allocate(idle_positions, unclaimed)

        assigned: Dict[str, Point] = {}
        for robot_id, position in proposed.items():
            frontier_id = self._frontier_id_for_position(active_tasks, position)
            if frontier_id is None:
                continue
            if await self.task_manager.claim_frontier(frontier_id, robot_id):
                assigned[robot_id] = position
                logger.info("allocated robot=%s -> frontier=%s at %s", robot_id, frontier_id, position)

        return assigned

    async def mark_frontier_reached(self, robot_id: str, robot_position: Point, frontier_position: Point) -> bool:
        
        distance = (
            (robot_position[0] - frontier_position[0]) ** 2
            + (robot_position[1] - frontier_position[1]) ** 2
        ) ** 0.5
        if distance > FRONTIER_REACHED_THRESHOLD_M:
            return False

        active_tasks = await self.task_manager.list_active_tasks()
        frontier_id = self._frontier_id_for_position(active_tasks, frontier_position)
        if frontier_id is not None:
            await self.task_manager.complete_frontier(frontier_id)
            logger.info("robot=%s completed frontier=%s", robot_id, frontier_id)
        return True

    @staticmethod
    def _frontier_id_for_position(tasks: List[dict], position: Point):
        for task in tasks:
            task_position = task.get("position")
            if task_position is not None and tuple(task_position) == tuple(position):
                return task["frontier_id"]
        return None
