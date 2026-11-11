import asyncio
import uuid
from datetime import datetime
from typing import Callable, Awaitable

# In-memory registry of active background scheduled tasks
SCHEDULED_JOBS: dict[str, dict] = {}
_running_tasks: dict[str, asyncio.Task] = {}

async def _job_loop(
    job_id: str,
    goal: str,
    interval_seconds: int,
    execute_fn: Callable[[str], Awaitable[None]]
) -> None:
    """Runs a multi-agent workflow repeatedly at the specified interval."""
    try:
        while job_id in SCHEDULED_JOBS:
            await asyncio.sleep(interval_seconds)
            if job_id not in SCHEDULED_JOBS:
                break
            SCHEDULED_JOBS[job_id]["runs"] += 1
            SCHEDULED_JOBS[job_id]["last_run"] = datetime.now().strftime("%I:%M:%S %p")
            await execute_fn(goal)
    except asyncio.CancelledError:
        pass

def register_scheduled_task(
    goal: str,
    interval_seconds: int,
    execute_fn: Callable[[str], Awaitable[None]]
) -> dict:
    """Registers and starts a background asyncio scheduled job."""
    job_id = str(uuid.uuid4())[:6]
    job_info = {
        "job_id": job_id,
        "goal": goal,
        "interval_seconds": interval_seconds,
        "runs": 0,
        "last_run": "Pending first run"
    }
    SCHEDULED_JOBS[job_id] = job_info
    task = asyncio.create_task(_job_loop(job_id, goal, interval_seconds, execute_fn))
    _running_tasks[job_id] = task
    return job_info

def cancel_all_jobs() -> int:
    """Stops all active background scheduled tasks."""
    count = len(_running_tasks)
    for task in _running_tasks.values():
        task.cancel()
    _running_tasks.clear()
    SCHEDULED_JOBS.clear()
    return count

def list_scheduled_jobs() -> list[dict]:
    """Returns all currently active scheduled tasks."""
    return list(SCHEDULED_JOBS.values())
