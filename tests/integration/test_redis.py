from __future__ import annotations

import pytest

pytest.importorskip("redis")

import asyncio  

from coordination.redis_client import RedisClient  
from coordination.state_manager import StateManager
from coordination.task_manager import TaskManager  
from robots.agents.robot_state import Pose, RobotState 

pytestmark = pytest.mark.integration


def _redis_available() -> bool:
    client = RedisClient()
    try:
        return asyncio.get_event_loop().run_until_complete(client.ping())
    except RuntimeError:
        
        return asyncio.run(client.ping())


@pytest.fixture(autouse=True)
def _skip_if_no_redis():
    if not _redis_available():
        pytest.skip("no reachable Redis instance - start one with `docker compose up redis` to run this file")


@pytest.fixture
def redis_client():
    client = RedisClient()
    yield client
    asyncio.run(client.close())


@pytest.mark.asyncio
async def test_state_manager_round_trips_robot_state(redis_client):
    state_manager = StateManager(redis_client)
    state = RobotState(robot_id="test_robot_01", pose=Pose(x=1.0, y=2.0, theta=0.5))

    await state_manager.update_state(state)
    retrieved = await state_manager.get_state("test_robot_01")

    assert retrieved is not None
    assert retrieved.pose.as_tuple() == (1.0, 2.0, 0.5)

    await state_manager.remove_state("test_robot_01")  

@pytest.mark.asyncio
async def test_state_manager_returns_none_for_unknown_robot(redis_client):
    state_manager = StateManager(redis_client)
    result = await state_manager.get_state("definitely_not_a_real_robot_id")
    assert result is None


@pytest.mark.asyncio
async def test_task_manager_claim_is_exclusive(redis_client):
    task_manager = TaskManager(redis_client)
    frontier_id = "test_frontier_exclusive_claim"

    await task_manager.register_frontier(frontier_id, (1.0, 1.0))
    first_claim = await task_manager.claim_frontier(frontier_id, "robot_a")
    second_claim = await task_manager.claim_frontier(frontier_id, "robot_b")

    assert first_claim is True
    assert second_claim is False  
    await task_manager.complete_frontier(frontier_id)  


@pytest.mark.asyncio
async def test_task_manager_release_only_by_current_holder(redis_client):
    task_manager = TaskManager(redis_client)
    frontier_id = "test_frontier_release_guard"

    await task_manager.register_frontier(frontier_id, (2.0, 2.0))
    await task_manager.claim_frontier(frontier_id, "robot_a")

    
    await task_manager.release_frontier(frontier_id, "robot_b")
    holder_after_wrong_release = await task_manager.get_claim_holder(frontier_id)
    assert holder_after_wrong_release == "robot_a"

    await task_manager.release_frontier(frontier_id, "robot_a")
    holder_after_correct_release = await task_manager.get_claim_holder(frontier_id)
    assert holder_after_correct_release is None


@pytest.mark.asyncio
async def test_task_manager_reassigns_orphaned_tasks(redis_client):
    task_manager = TaskManager(redis_client)
    frontier_id = "test_frontier_orphaned"

    await task_manager.register_frontier(frontier_id, (3.0, 3.0))
    await task_manager.claim_frontier(frontier_id, "failed_robot")

    released = await task_manager.reassign_orphaned_tasks("failed_robot")

    assert frontier_id in released
    assert await task_manager.get_claim_holder(frontier_id) is None
