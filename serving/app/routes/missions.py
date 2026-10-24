from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException

from coordination.mission_manager import MissionManager
from serving.app.dependencies import get_mission_manager
from serving.app.schemas import (
    MissionInfoResponse,
    MissionStartRequest,
    MissionStartResponse,
    RobotFailureReport,
)

router = APIRouter()


@router.post("/missions", response_model=MissionStartResponse)
async def start_mission(
    request: MissionStartRequest,
    mission_manager: MissionManager = Depends(get_mission_manager),
) -> MissionStartResponse:
    mission_id = uuid.uuid4().hex[:12]
    info = await mission_manager.start_mission(mission_id, request.robot_ids)
    return MissionStartResponse(mission_id=info.mission_id, status=info.status.value, robot_ids=info.robot_ids)


@router.get("/missions/{mission_id}", response_model=MissionInfoResponse)
async def get_mission(
    mission_id: str,
    mission_manager: MissionManager = Depends(get_mission_manager),
) -> MissionInfoResponse:
    info = await mission_manager.get_mission_info(mission_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"no mission found with mission_id={mission_id!r}")

    return MissionInfoResponse(
        mission_id=info.mission_id,
        status=info.status.value,
        robot_ids=info.robot_ids,
        failed_robot_ids=info.failed_robot_ids,
    )


@router.post("/missions/{mission_id}/failure")
async def report_robot_failure(
    mission_id: str,
    report: RobotFailureReport,
    mission_manager: MissionManager = Depends(get_mission_manager),
) -> dict:
    released_tasks = await mission_manager.handle_robot_failure(mission_id, report.robot_id)
    return {"robot_id": report.robot_id, "released_tasks": released_tasks}
