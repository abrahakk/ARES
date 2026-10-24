from __future__ import annotations

import enum
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from coordination.event_bus import EventBus
from coordination.redis_client import RedisClient
from coordination.state_manager import StateManager
from coordination.task_manager import TaskManager
from robots.agents.robot_state import RobotState, RobotStatus
from robots.communication.message import Message, MessageType

logger = logging.getLogger(__name__)


class MissionStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class MissionInfo:
    mission_id: str
    robot_ids: List[str]
    status: MissionStatus = MissionStatus.PENDING
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    failed_robot_ids: List[str] = field(default_factory=list)


def _mission_key(mission_id: str) -> str:
    return f"mission:{mission_id}:info"


class MissionManager:
    def __init__(self, redis_client: RedisClient) -> None:
        self.client = redis_client
        self.state_manager = StateManager(redis_client)
        self.event_bus = EventBus(redis_client)
        self.task_manager = TaskManager(redis_client)



    async def start_mission(self, mission_id: str, robot_ids: List[str]) -> MissionInfo:
        info = MissionInfo(mission_id=mission_id, robot_ids=robot_ids, status=MissionStatus.RUNNING, started_at=time.time())
        await self._save_mission_info(info)
        await self._publish_mission_event(mission_id, "mission_started", {"robot_ids": robot_ids})
        logger.info("mission=%s started with robots=%s", mission_id, robot_ids)
        return info

    async def end_mission(self, mission_id: str, status: MissionStatus) -> None:
        info = await self.get_mission_info(mission_id)
        if info is None:
            logger.warning("end_mission: unknown mission_id=%s", mission_id)
            return
        info.status = status
        info.ended_at = time.time()
        await self._save_mission_info(info)
        await self._publish_mission_event(mission_id, "mission_ended", {"status": status.value})
        logger.info("mission=%s ended with status=%s", mission_id, status.value)

    async def get_mission_info(self, mission_id: str) -> Optional[MissionInfo]:
        if self.client.raw is None:
            return None
        raw = await self.client.raw.get(_mission_key(mission_id))
        if raw is None:
            return None
        data = json.loads(raw)
        data["status"] = MissionStatus(data["status"])
        return MissionInfo(**data)

    async def _save_mission_info(self, info: MissionInfo) -> None:
        if self.client.raw is None:
            logger.debug("stub mode: would save mission info for %s", info.mission_id)
            return
        payload = {
            "mission_id": info.mission_id,
            "robot_ids": info.robot_ids,
            "status": info.status.value,
            "started_at": info.started_at,
            "ended_at": info.ended_at,
            "failed_robot_ids": info.failed_robot_ids,
        }
        await self.client.raw.set(_mission_key(info.mission_id), json.dumps(payload))



    async def handle_robot_failure(self, mission_id: str, robot_id: str) -> List[str]:
        
        state = await self.state_manager.get_state(robot_id)
        if state is not None:
            state.status = RobotStatus.FAILED
            await self.state_manager.update_state(state)

        released_tasks = await self.task_manager.reassign_orphaned_tasks(robot_id)

        info = await self.get_mission_info(mission_id)
        if info is not None and robot_id not in info.failed_robot_ids:
            info.failed_robot_ids.append(robot_id)
            await self._save_mission_info(info)

        await self.event_bus.publish(
            Message(
                sender_id="mission_manager",
                message_type=MessageType.EMERGENCY,
                payload={"reason": "robot_failure", "robot_id": robot_id},
            )
        )

        logger.info(
            "handled failure for robot=%s in mission=%s, released %d task(s)",
            robot_id, mission_id, len(released_tasks),
        )
        return released_tasks


    async def get_active_robot_states(self, robot_ids: List[str]) -> Dict[str, RobotState]:
        all_states = await self.state_manager.get_all_states(robot_ids)
        return {
            robot_id: state
            for robot_id, state in all_states.items()
            if state.status != RobotStatus.FAILED
        }

    async def _publish_mission_event(self, mission_id: str, event_type: str, payload: dict) -> None:
        if self.client.raw is None:
            logger.debug("stub mode: would publish mission event %s for %s", event_type, mission_id)
            return
        event = {"mission_id": mission_id, "event_type": event_type, "payload": payload, "timestamp": time.time()}
        await self.client.raw.publish(self.client.settings.mission_events_channel, json.dumps(event))
