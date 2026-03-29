"""Run coroutines from synchronous sampler code.

This module centralizes the event-loop bridging used by sampler execution
paths so synchronous orchestration can safely invoke async helpers in both
normal scripts and already-running event loops.
"""

import asyncio
import threading
from typing import Any, Callable


def run_coroutine_from_sync(coroutine_factory: Callable[[], Any]) -> Any:
    """Run an async workflow from synchronous code.

    This helper executes the coroutine directly when no event loop is active.
    If the caller already runs inside an event loop, it executes the coroutine
    in a dedicated worker thread and re-raises any exception from that thread.

    Args:
        coroutine_factory (Callable[[], Any]): Factory returning the coroutine
            to execute.

    Returns:
        Any: The value returned by the coroutine.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine_factory())

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coroutine_factory())
        except BaseException as exc:  # pragma: no cover - passthrough
            error["value"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "value" in error:
        raise error["value"]
    return result["value"]
