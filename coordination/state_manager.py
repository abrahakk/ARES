from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from coordination.redis_client import RedisClient
from robots.agents.robot_state import RobotState

logger = logging.getLogger(__name__)


class StateManager:
    def __init__(self, redis_client: RedisClient) -> None:
        self.client = redis_client

    def _key(self, robot_id: str) -> str:
        return f"robot:{robot_id}:state"

    async def update_state(self, state: RobotState) -> None:
        state.touch()
        if self.client.raw is None:
            logger.debug("stub mode: would update state for robot %s", state.robot_id)
            return
        await self.client.raw.set(
            self._key(state.robot_id),
            json.dumps(state.to_dict()),
            ex=self.client.settings.key_ttl_s,
        )

    async def get_state(self, robot_id: str) -> Optional[RobotState]:
        if self.client.raw is None:
            return None
        raw = await self.client.raw.get(self._key(robot_id))
        if raw is None:
            return None
        return RobotState.from_dict(json.loads(raw))

    async def get_all_states(self, robot_ids: List[str]) -> Dict[str, RobotState]:
  
        states: Dict[str, RobotState] = {}
        for robot_id in robot_ids:
            state = await self.get_state(robot_id)
            if state is not None:
                states[robot_id] = state
        return states

    async def list_active_robot_ids(self, pattern: str = "robot:*:state") -> List[str]:
      
        if self.client.raw is None:
            return []
        robot_ids = []
        async for key in self.client.raw.scan_iter(match=pattern):
      
            parts = key.split(":")
            if len(parts) == 3:
                robot_ids.append(parts[1])
        return robot_ids

    async def remove_state(self, robot_id: str) -> None:
        if self.client.raw is None:
            return
        await self.client.raw.delete(self._key(robot_id))
