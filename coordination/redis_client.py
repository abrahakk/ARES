from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as redis
except ImportError:  
    redis = None
    logger.warning("redis package not installed - RedisClient will run in stub mode.")


@dataclass
class RedisSettings:
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    socket_timeout_s: float = 2.0
    key_ttl_s: int = 30
    messages_channel: str = "ares:messages"
    mission_events_channel: str = "ares:mission"

    @classmethod
    def from_yaml(cls, path: str = "config/system.yaml") -> "RedisSettings":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        redis_cfg = data.get("redis", {})
        channels = redis_cfg.get("channels", {})
        return cls(
            host=redis_cfg.get("host", cls.host),
            port=redis_cfg.get("port", cls.port),
            db=redis_cfg.get("db", cls.db),
            socket_timeout_s=redis_cfg.get("socket_timeout_s", cls.socket_timeout_s),
            key_ttl_s=redis_cfg.get("key_ttl_s", cls.key_ttl_s),
            messages_channel=channels.get("messages", cls.messages_channel),
            mission_events_channel=channels.get("mission_events", cls.mission_events_channel),
        )


class RedisClient:
    def __init__(self, settings: Optional[RedisSettings] = None) -> None:
        self.settings = settings or RedisSettings.from_yaml()
        self.raw = None
        if redis is not None:
            self.raw = redis.Redis(
                host=self.settings.host,
                port=self.settings.port,
                db=self.settings.db,
                socket_timeout=self.settings.socket_timeout_s,
                decode_responses=True,
            )
        else:
            logger.debug("RedisClient constructed in stub mode - no redis package available")

    async def ping(self) -> bool:
        if self.raw is None:
            return False
        try:
            return await self.raw.ping()
        except Exception as exc:  
            logger.warning("Redis ping failed: %s", exc)
            return False

    async def close(self) -> None:
        if self.raw is not None:
            await self.raw.close()
