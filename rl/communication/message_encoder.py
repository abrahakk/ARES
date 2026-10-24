from __future__ import annotations

from typing import List

import numpy as np

from robots.communication.message import Message, MessageType

MESSAGE_TYPES: List[MessageType] = list(MessageType)
TYPE_VECTOR_DIM = len(MESSAGE_TYPES)
PAYLOAD_VECTOR_DIM = 4  
ENCODED_MESSAGE_DIM = TYPE_VECTOR_DIM + PAYLOAD_VECTOR_DIM


class MessageEncoder:
    @staticmethod
    def encode(message: Message) -> np.ndarray:
        type_one_hot = np.zeros(TYPE_VECTOR_DIM, dtype=np.float32)
        type_one_hot[MESSAGE_TYPES.index(message.message_type)] = 1.0

        payload = message.payload or {}
        payload_vec = np.array(
            [
                float(payload.get("x", 0.0)),
                float(payload.get("y", 0.0)),
                float(payload.get("priority", 0.0)),
                float(payload.get("value", 0.0)),
            ],
            dtype=np.float32,
        )
        return np.concatenate([type_one_hot, payload_vec])

    @staticmethod
    def encode_batch(messages: List[Message]) -> np.ndarray:
        if not messages:
            return np.zeros((0, ENCODED_MESSAGE_DIM), dtype=np.float32)
        return np.stack([MessageEncoder.encode(m) for m in messages])

    @staticmethod
    def decode_payload(message_type: MessageType, payload_vector: np.ndarray) -> dict:
        
       x, y, priority, value = payload_vector[:4].tolist()
        return {"x": x, "y": y, "priority": priority, "value": value}
