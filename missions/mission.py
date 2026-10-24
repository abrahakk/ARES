from __future__ import annotations

import enum
import logging
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from coordination.mission_manager import MissionInfo, MissionManager, MissionStatus
from coordination.redis_client import RedisClient
from simulation.scenarios.exploration import ExplorationScenario

logger = logging.getLogger(__name__)


class MissionState(str, enum.Enum):


    PLANNING = "planning"
    RUNNING = "running"
    COMPLETING = "completing"
    ENDED = "ended"


@dataclass
class Mission:
    scenario: ExplorationScenario
    mission_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: MissionState = MissionState.PLANNING

    def __post_init__(self) -> None:
        self._redis_client: Optional[RedisClient] = None
        self._mission_manager: Optional[MissionManager] = None

    @property
    def robot_ids(self) -> List[str]:
        return self.scenario.robot_ids

    def _ensure_manager(self) -> MissionManager:
        if self._mission_manager is None:
            self._redis_client = RedisClient()
            self._mission_manager = MissionManager(self._redis_client)
        return self._mission_manager

    async def start(self) -> MissionInfo:
        manager = self._ensure_manager()
        info = await manager.start_mission(self.mission_id, self.robot_ids)
        self.state = MissionState.RUNNING
        logger.info("mission=%s ('%s') started with %d robots", self.mission_id, self.scenario.name, len(self.robot_ids))
        return info

    async def end(self, status: MissionStatus = MissionStatus.COMPLETED) -> None:
        self.state = MissionState.COMPLETING
        manager = self._ensure_manager()
        await manager.end_mission(self.mission_id, status)
        self.state = MissionState.ENDED
        logger.info("mission=%s ended with status=%s", self.mission_id, status.value)

    async def get_info(self) -> Optional[MissionInfo]:
        manager = self._ensure_manager()
        return await manager.get_mission_info(self.mission_id)

    async def report_robot_failure(self, robot_id: str) -> List[str]:

        manager = self._ensure_manager()
        return await manager.handle_robot_failure(self.mission_id, robot_id)

    async def close(self) -> None:
        if self._redis_client is not None:
            await self._redis_client.close()
