from __future__ import annotations

import json
import logging
from typing import List, Optional, Tuple

from coordination.redis_client import RedisClient

logger = logging.getLogger(__name__)

Point = Tuple[float, float]


def _frontier_key(frontier_id: str) -> str:
    return f"task:frontier:{frontier_id}"


def _active_tasks_key() -> str:
    return "mission:active_tasks"


class TaskManager:
    def __init__(self, redis_client: RedisClient, claim_ttl_s: int = 60) -> None:
        self.client = redis_client
        self.claim_ttl_s = claim_ttl_s

    async def register_frontier(self, frontier_id: str, position: Point) -> None:

        if self.client.raw is None:
            logger.debug("stub mode: would register frontier %s at %s", frontier_id, position)
            return
        await self.client.raw.hset(
            _active_tasks_key(), frontier_id, json.dumps({"position": position, "claimed_by": None})
        )

    async def claim_frontier(self, frontier_id: str, robot_id: str) -> bool:

        if self.client.raw is None:
            logger.debug("stub mode: would claim frontier %s for robot %s", frontier_id, robot_id)
            return True

        claimed = await self.client.raw.set(
            _frontier_key(frontier_id), robot_id, nx=True, ex=self.claim_ttl_s
        )
        success = bool(claimed)
        if success:
            await self.client.raw.hset(
                _active_tasks_key(), frontier_id, json.dumps({"claimed_by": robot_id})
            )
            logger.debug("robot=%s claimed frontier=%s", robot_id, frontier_id)
        return success

    async def release_frontier(self, frontier_id: str, robot_id: str) -> None:
        if self.client.raw is None:
            logger.debug("stub mode: would release frontier %s from robot %s", frontier_id, robot_id)
            return

        current_holder = await self.client.raw.get(_frontier_key(frontier_id))
        if current_holder == robot_id:
            await self.client.raw.delete(_frontier_key(frontier_id))
            await self.client.raw.hdel(_active_tasks_key(), frontier_id)

    async def complete_frontier(self, frontier_id: str) -> None:

        if self.client.raw is None:
            logger.debug("stub mode: would complete frontier %s", frontier_id)
            return
        await self.client.raw.delete(_frontier_key(frontier_id))
        await self.client.raw.hdel(_active_tasks_key(), frontier_id)

    async def get_claim_holder(self, frontier_id: str) -> Optional[str]:
        if self.client.raw is None:
            return None
        return await self.client.raw.get(_frontier_key(frontier_id))

    async def list_active_tasks(self) -> List[dict]:

        if self.client.raw is None:
            return []
        raw_tasks = await self.client.raw.hgetall(_active_tasks_key())
        tasks = []
        for frontier_id, raw_value in raw_tasks.items():
            data = json.loads(raw_value)
            data["frontier_id"] = frontier_id
            tasks.append(data)
        return tasks

    async def reassign_orphaned_tasks(self, failed_robot_id: str) -> List[str]:

        released = []
        for task in await self.list_active_tasks():
            if task.get("claimed_by") == failed_robot_id:
                frontier_id = task["frontier_id"]
                await self.release_frontier(frontier_id, failed_robot_id)
                released.append(frontier_id)
        if released:
            logger.info("reassigned %d orphaned task(s) from failed robot=%s", len(released), failed_robot_id)
        return released
