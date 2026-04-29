import json
import os
from typing import Any, Optional

import redis


class RedisCache:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.key_prefix = os.getenv("REDIS_KEY_PREFIX", "supermew")
        self.default_ttl = int(os.getenv("REDIS_CACHE_TTL_SECONDS", "300"))
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def _key(self, key: str) -> str:
        return f"{self.key_prefix}:{key}"

    def get_json(self, key: str) -> Optional[Any]:
        try:
            value = self._get_client().get(self._key(key))
            if not value:
                return None
            return json.loads(value)
        except Exception:
            return None

    def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        try:
            payload = json.dumps(value, ensure_ascii=False)
            ttl_val = ttl if ttl is not None else self.default_ttl
            if ttl_val <= 0:
                self._get_client().set(self._key(key), payload)
            else:
                self._get_client().setex(self._key(key), ttl_val, payload)
        except Exception:
            return

    def delete(self, key: str) -> None:
        try:
            self._get_client().delete(self._key(key))
        except Exception:
            return

    def delete_pattern(self, pattern: str) -> None:
        """使用 SCAN 替代 KEYS，避免 Redis 阻塞"""
        try:
            full_pattern = self._key(pattern)
            client = self._get_client()
            cursor = 0
            while True:
                cursor, keys = client.scan(cursor=cursor, match=full_pattern, count=100)
                if keys:
                    client.delete(*keys)
                if cursor == 0:
                    break
        except Exception:
            return


cache = RedisCache()
