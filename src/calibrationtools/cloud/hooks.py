from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .task_payload import CloudTaskContext, ResolvedSharedAsset


@dataclass(frozen=True)
class CloudSessionContext:
    session_id: str
    input_mount_path: str
    output_mount_path: str
    logs_mount_path: str
    shared_assets: tuple[ResolvedSharedAsset, ...] = ()


class CloudRunnerHooks:
    """Optional public extension hooks for model-specific cloud behavior."""

    def prepare_session(self, context: CloudSessionContext) -> None:
        """Inspect or validate a session before Azure resources are created."""

    def prepare_task_payload(
        self,
        payload: dict[str, Any],
        context: CloudTaskContext,
    ) -> dict[str, Any]:
        """Return the final task payload after declarative transforms."""
        return payload

    def read_output_dir(
        self,
        output_dir: Path,
        context: CloudTaskContext,
    ) -> Any:
        """Optionally override output parsing for one completed task."""
        return NotImplemented
