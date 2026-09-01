"""Azure Batch pool and node health checks.

Batch surfaces pool problems (image pull failures, start-task errors,
quota-blocked resizes) on the nodes and the pool rather than on the job, so a
job that never starts otherwise looks identical to a job that is merely
slow. These functions turn that node-level state into progress messages and
fail-fast errors, for both initial pool allocation and later job polling.
"""

from __future__ import annotations

import time
from typing import Any, Callable

#: Node states from which a pool can never make progress on queued tasks.
_UNUSABLE_NODE_STATES = frozenset(
    {"unusable", "starttaskfailed", "preempted", "offline", "unknown"}
)

#: Node states in which a node can accept or is already running task work.
_USABLE_NODE_STATES = frozenset({"idle", "running", "leavingpool"})


def state_summary(states: dict[str, int]) -> str:
    """Render node state counts in a stable, compact order."""

    return ", ".join(
        f"{name} {count}" for name, count in sorted(states.items())
    )


def state_name(state: Any) -> str:
    """Normalize an Azure Batch state enum or string to a plain name."""

    value = getattr(state, "value", state)
    return str(value).rsplit(".", 1)[-1].replace("_", "").lower()


def error_text(error: Any) -> str:
    """Render an Azure Batch error object as one compact line."""

    code = getattr(error, "code", None)
    message = getattr(error, "message", None)
    message = getattr(message, "value", message)
    text = " ".join(str(part) for part in (code, message) if part)
    return " ".join(text.split()) or str(error)


def list_compute_nodes(batch_client: Any, pool_name: str) -> list[Any]:
    """List pool nodes across supported Azure Batch client versions."""

    if hasattr(batch_client, "list_compute_nodes"):
        return list(batch_client.list_compute_nodes(pool_name))
    return list(batch_client.compute_node.list(pool_name))


def pool_resize_errors(batch_client: Any, pool_name: str) -> list[str]:
    """Return pool resize errors, which usually indicate quota problems."""

    try:
        if hasattr(batch_client, "get_pool"):
            pool = batch_client.get_pool(pool_name)
        else:
            pool = batch_client.pool.get(pool_name)
    except Exception:
        return []
    return [
        error_text(error)
        for error in getattr(pool, "resize_errors", None) or []
    ]


def pool_health(cloud_client: Any, pool_name: str) -> dict[str, Any]:
    """Summarize compute-node states and errors for one Batch pool.

    Args:
        cloud_client (Any): Client used to inspect the pool's nodes.
        pool_name (str): Name of the pool to summarize.

    Returns:
        dict[str, Any]: Node state counts plus any node or resize errors.
        Keys are omitted when the client does not expose that information.
    """

    batch_client = getattr(cloud_client, "batch_service_client", None)
    if batch_client is None:
        return {}
    try:
        nodes = list_compute_nodes(batch_client, pool_name)
    except Exception:
        return {}
    states: dict[str, int] = {}
    errors: list[str] = []
    for node in nodes:
        state = state_name(getattr(node, "state", None))
        states[state] = states.get(state, 0) + 1
        for error in getattr(node, "errors", None) or []:
            errors.append(error_text(error))
        start_task_info = getattr(node, "start_task_info", None)
        failure_info = getattr(start_task_info, "failure_info", None)
        if failure_info is not None:
            errors.append(f"start task: {error_text(failure_info)}")
    health: dict[str, Any] = {"node_states": states}
    if errors:
        health["node_errors"] = sorted(set(errors))[:5]
    resize_errors = pool_resize_errors(batch_client, pool_name)
    if resize_errors:
        health["resize_errors"] = resize_errors
    return health


def raise_for_unhealthy_pool(
    pool_name: str, context: str, health: dict[str, Any]
) -> None:
    """Fail fast when every allocated node is in a terminal bad state.

    Args:
        pool_name (str): Name of the pool being evaluated.
        context (str): Phase description included in the error message.
        health (dict[str, Any]): Result of :func:`pool_health`.

    Returns:
        None: This function does not return a value.

    Raises:
        RuntimeError: If no allocated node can ever run a task.
    """

    states = health.get("node_states") or {}
    if not states or not set(states) <= _UNUSABLE_NODE_STATES:
        return
    detail = "; ".join(
        list(health.get("node_errors", ()))
        + list(health.get("resize_errors", ()))
    )
    message = (
        f"Azure Batch pool {pool_name} has no usable nodes during "
        f"{context} (node states: {state_summary(states)})."
    )
    if detail:
        message += f" Details: {detail}"
    raise RuntimeError(message)


def _node_wait_message(
    pool_name: str, states: dict[str, int], usable: int, elapsed: float
) -> str:
    """Summarize node allocation progress for one status line."""

    if not states:
        return (
            f"Waiting for pool {pool_name} to allocate nodes ({elapsed:.0f}s)"
        )
    summary = state_summary(states)
    if usable:
        return f"Pool {pool_name} nodes ready ({summary})"
    return f"Waiting for pool {pool_name} nodes: {summary} ({elapsed:.0f}s)"


def wait_for_pool_nodes(
    cloud_client: Any,
    *,
    pool_name: str,
    wait_for_nodes: bool,
    poll_interval: float,
    max_wait_for_nodes: float,
    emit: Callable[..., None],
) -> None:
    """Poll until the pool allocates at least one usable compute node.

    ``create_pool`` returns as soon as Batch accepts the pool definition,
    long before any VM boots, pulls the container image, and runs its start
    task. Polling node states here turns the silent gap between pool
    creation and first task execution into visible progress, and surfaces
    allocation failures instead of letting the job wait for them.

    Args:
        cloud_client (Any): Client used to inspect the pool's nodes.
        pool_name (str): Name of the pool to poll.
        wait_for_nodes (bool): Whether to poll at all; disabling returns
            immediately for callers that skip this check.
        poll_interval (float): Seconds to sleep between polls.
        max_wait_for_nodes (float): Maximum seconds to wait for a usable
            node before raising.
        emit (Callable[..., None]): Progress-message sink.

    Returns:
        None: This function does not return a value.

    Raises:
        TimeoutError: If no usable node appears within
            ``max_wait_for_nodes``.
    """

    if not wait_for_nodes:
        return
    started = time.monotonic()
    while True:
        health = pool_health(cloud_client, pool_name)
        states = health.get("node_states")
        if states is None:
            return
        usable = sum(
            count
            for name, count in states.items()
            if name in _USABLE_NODE_STATES
        )
        elapsed = time.monotonic() - started
        emit(
            _node_wait_message(pool_name, states, usable, elapsed),
            stage="pool_nodes",
            pool_name=pool_name,
            usable_nodes=usable,
            elapsed_seconds=elapsed,
            **health,
        )
        if usable:
            return
        raise_for_unhealthy_pool(
            pool_name, f"pool {pool_name} startup", health
        )
        if elapsed > max_wait_for_nodes:
            detail = "; ".join(
                list(health.get("node_errors", ()))
                + list(health.get("resize_errors", ()))
            )
            message = (
                f"Azure Batch pool {pool_name} allocated no usable nodes "
                f"within {max_wait_for_nodes:.0f}s (node states: "
                f"{state_summary(states)}). This usually means a "
                "quota-blocked resize, an image pull failure, or a failing "
                "start task."
            )
            if detail:
                message += f" Details: {detail}"
            raise TimeoutError(message)
        time.sleep(poll_interval)
