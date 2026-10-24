from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from serving.app.dependencies import get_model_manager
from serving.app.schemas import ActionResponse, Observation
from serving.inference.model_manager import ModelManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/predict", response_model=ActionResponse)
async def predict(
    observation: Observation,
    model_manager: ModelManager = Depends(get_model_manager),
) -> ActionResponse:
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="model not loaded")

    try:
        result = model_manager.predict(observation.model_dump())
    except Exception as exc:  
        logger.exception("inference failed for robot_id=%s", observation.robot_id)
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc

    return ActionResponse(
        robot_id=observation.robot_id,
        linear_velocity=result["linear_velocity"],
        angular_velocity=result["angular_velocity"],
        should_communicate=result["should_communicate"],
        value_estimate=result.get("value_estimate"),
    )

