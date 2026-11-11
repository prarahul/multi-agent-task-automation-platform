import os
import json
import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_fallback_memory: list[str] = []

try:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=1)
    redis_client.ping()
    REDIS_ACTIVE = True
except Exception:
    redis_client = None
    REDIS_ACTIVE = False

def append_chat_memory(session_id: str, user_msg: str, ai_msg: str) -> None:
    """Stores conversational turns in Redis list (or local fallback) and publishes telemetry."""
    entry_user = f"User: {user_msg}"
    entry_ai = f"Assistant: {ai_msg}"
    
    if REDIS_ACTIVE and redis_client is not None:
        key = f"platform:chat_history:{session_id}"
        redis_client.rpush(key, entry_user, entry_ai)
        redis_client.ltrim(key, -10, -1)
        redis_client.publish(f"platform:events:{session_id}", json.dumps({"event": "MEMORY_UPDATED"}))
    else:
        _fallback_memory.extend([entry_user, entry_ai])
        if len(_fallback_memory) > 10:
            del _fallback_memory[:-10]

def get_chat_memory(session_id: str, limit: int = 6) -> list[str]:
    """Retrieves recent conversation context from Redis."""
    if REDIS_ACTIVE and redis_client is not None:
        key = f"platform:chat_history:{session_id}"
        raw_items = redis_client.lrange(key, -limit, -1)
        return [str(item) for item in raw_items]
    return _fallback_memory[-limit:]
