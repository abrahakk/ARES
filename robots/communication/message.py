from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Optional


class MessageType(str, Enum):
    POSE_UPDATE = "pose_update"
    FRONTIER_DISCOVERED = "frontier_discovered"
    FRONTIER_CLAIMED = "frontier_claimed"
    TASK_COMPLETE = "task_complete"
    OBSTACLE_REPORT = "obstacle_report"
    EMERGENCY = "emergency"
    HEARTBEAT = "heartbeat"


@dataclass
class Message:
    sender_id: str
    message_type: MessageType
    payload: Dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    recipient_id: Optional[str] = None

    @property
    def size_bytes(self) -> int:
        
        return len(json.dumps(self.to_dict()).encode("utf-8"))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["message_type"] = self.message_type.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        data = dict(data)
        if "message_type" in data and not isinstance(data["message_type"], MessageType):
            data["message_type"] = MessageType(data["message_type"])
        return cls(**data)

