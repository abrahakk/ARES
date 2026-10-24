from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from robots.communication.message import Message, MessageType

logger = logging.getLogger(__name__)


class CommunicationManager:
    def __init__(self, robot_id: str) -> None:
        self.robot_id = robot_id
        self._send_hook: Optional[Callable[[Message], None]] = None

    def attach_send_hook(self, send_hook: Callable[[Message], None]) -> None:
        
        self._send_hook = send_hook

    def build_message(
        self,
        payload: Dict[str, Any],
        message_type: MessageType = MessageType.FRONTIER_DISCOVERED,
        recipient_id: Optional[str] = None,
    ) -> Message:
        return Message(
            sender_id=self.robot_id,
            message_type=message_type,
            payload=payload,
            recipient_id=recipient_id,
        )

    def send(self, message: Message) -> None:
        if self._send_hook is None:
            logger.warning(
                "robot=%s tried to send a message with no send hook attached "
                "(call attach_send_hook() during setup)",
                self.robot_id,
            )
            return
        self._send_hook(message)

    def send_heartbeat(self) -> None:
        self.send(self.build_message(payload={}, message_type=MessageType.HEARTBEAT))

    def send_emergency(self, reason: str) -> None:
        self.send(
            self.build_message(
                payload={"reason": reason},
                message_type=MessageType.EMERGENCY,
            )
        )
