from __future__ import annotations

import asyncio
import json
import logging
from typing import Awaitable, Callable, Optional

from coordination.redis_client import RedisClient
from robots.communication.message import Message

logger = logging.getLogger(__name__)

MessageHandler = Callable[[Message], Awaitable[None]]


class EventBus:
    def __init__(self, redis_client: RedisClient) -> None:
        self.client = redis_client
        self._listen_task: Optional[asyncio.Task] = None
        self._pubsub = None

    async def publish(self, message: Message) -> None:
        if self.client.raw is None:
            logger.debug("stub mode: would publish message %s", message.message_id)
            return
        await self.client.raw.publish(
            self.client.settings.messages_channel, json.dumps(message.to_dict())
        )

    async def start_listening(self, handler: MessageHandler) -> None:
        
        if self.client.raw is None:
            logger.debug("stub mode: start_listening is a no-op without a real Redis connection")
            return

        self._pubsub = self.client.raw.pubsub()
        await self._pubsub.subscribe(self.client.settings.messages_channel)
        self._listen_task = asyncio.create_task(self._listen_loop(handler))

    async def _listen_loop(self, handler: MessageHandler) -> None:
        assert self._pubsub is not None
        try:
            async for raw_message in self._pubsub.listen():
                if raw_message.get("type") != "message":
                    continue
                try:
                    payload = json.loads(raw_message["data"])
                    message = Message.from_dict(payload)
                except (json.JSONDecodeError, TypeError, KeyError) as exc:
                    logger.warning("dropping malformed pub/sub payload: %s", exc)
                    continue
                await handler(message)
        except asyncio.CancelledError:
            logger.debug("event bus listen loop cancelled")
            raise

    async def stop_listening(self) -> None:
        if self._listen_task is not None:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None
        if self._pubsub is not None:
            await self._pubsub.unsubscribe(self.client.settings.messages_channel)
            await self._pubsub.close()
            self._pubsub = None
