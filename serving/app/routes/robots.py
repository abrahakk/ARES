from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException

from coordination.state_manager import StateManager
from serving.app.dependencies import get_state_manager
from serving.app.schemas import RobotPose, RobotStateResponse

router = APIRouter()


@router.get("/robots", response_model=List[str])
async def list_active_robots(state_manager: StateManager = Depends(get_state_manager)) -> List[str]:
    return await state_manager.list_active_robot_ids()


@router.get("/robots/{robot_id}", response_model=RobotStateResponse)
async def get_robot_state(
    robot_id: str,
    state_manager: StateManager = Depends(get_state_manager),
) -> RobotStateResponse:
    state = await state_manager.get_state(robot_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"no state found for robot_id={robot_id!r} (unknown or expired)")

    return RobotStateResponse(
        robot_id=state.robot_id,
        pose=RobotPose(x=state.pose.x, y=state.pose.y, theta=state.pose.theta),
        status=state.status.value,
        battery_level=state.battery_level,
        last_updated=state.last_updated,
        collision_count=state.collision_count,
    )
