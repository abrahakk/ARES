from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from robots.communication.message import Message

logger = logging.getLogger(__name__)


@dataclass
class ProtocolConfig:
    max_bandwidth_bytes_per_step: int = 2048
    packet_loss_probability: float = 0.0  

    max_range_meters: float = 15.0
    latency_steps: int = 0                

class CommunicationProtocol:
    
    def __init__(self, config: ProtocolConfig | None = None) -> None:
        self.config = config or ProtocolConfig()
        self._bandwidth_used_this_step: int = 0

    def reset_step(self) -> None:
        
        self._bandwidth_used_this_step = 0

    def should_deliver(
        self,
        message: Message,
        sender_pos: tuple,
        recipient_pos: tuple,
    ) -> bool:
        
        if self._out_of_range(sender_pos, recipient_pos):
            return False

        if self._bandwidth_used_this_step + message.size_bytes > self.config.max_bandwidth_bytes_per_step:
            logger.debug("message %s dropped: bandwidth budget exceeded", message.message_id)
            return False

        if random.random() < self.config.packet_loss_probability:
            logger.debug("message %s dropped: simulated packet loss", message.message_id)
            return False

        self._bandwidth_used_this_step += message.size_bytes
        return True

    def _out_of_range(self, sender_pos: tuple, recipient_pos: tuple) -> bool:
        dx = sender_pos[0] - recipient_pos[0]
        dy = sender_pos[1] - recipient_pos[1]
        distance = (dx ** 2 + dy ** 2) ** 0.5
        return distance > self.config.max_range_meters
