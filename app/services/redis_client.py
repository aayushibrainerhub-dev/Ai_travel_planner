import os
from typing import Optional

from redis import Redis


def get_redis_url() -> str:
    return os.getenv("REDIS_URL", "redis://localhost:6379/0")


def get_client() -> Redis:
    url = get_redis_url()
    return Redis.from_url(url, decode_responses=False)


def set_key(key: str, value: str, ex: Optional[int] = None) -> bool:
    r = get_client()
    print("r--------------------------", r)
    return r.set(name=key, value=value, ex=ex)


def get_key(key: str) -> Optional[str]:
    r = get_client()
    return r.get(key)
