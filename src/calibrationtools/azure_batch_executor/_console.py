"""Console output helpers for Azure Batch operations.

``cfa-cloudops`` writes advisories and per-file progress with bare ``print``
calls, its own log handlers, and ``tqdm`` progress bars. Those writes land
mid-frame in the sampler's live display and force partial redraws, which is
why a running calibration otherwise appears to emit duplicated, truncated
progress bars.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from typing import Any, Iterator


@contextmanager
def capture_cloudops_output(enabled: bool = True) -> Iterator[list[str]]:
    """Divert ``cfa-cloudops`` console output into a list for re-reporting.

    Args:
        enabled (bool): Whether to capture output. Disabling restores the
            default passthrough behavior for debugging.

    Yields:
        list[str]: Captured lines, appended as they are produced by logging
            and populated from stdout once the block exits.
    """

    if not enabled:
        yield []
    else:
        lines: list[str] = []

        class _Collector(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                lines.append(record.getMessage().strip())

        buffer = StringIO()
        logger = logging.getLogger("cfa")
        handler = _Collector()
        previous_propagate = logger.propagate
        logger.addHandler(handler)
        logger.propagate = False
        try:
            with redirect_stdout(buffer):
                yield lines
        finally:
            logger.removeHandler(handler)
            logger.propagate = previous_propagate
            lines.extend(
                line.strip()
                for line in buffer.getvalue().splitlines()
                if line.strip()
            )


@contextmanager
def suppress_upload_progress_bar() -> Iterator[None]:
    """Silence ``cfa-cloudops``'s own ``tqdm``-based upload progress bar.

    ``cfa-cloudops`` drives its upload loop with ``tqdm``, which draws its
    own terminal region. Whoever calls this executor already owns the one
    live display for the process (a standalone run's bar or a study's
    dashboard), so a second independent renderer here would fight it for
    the same terminal lines instead of interleaving cleanly. The caller
    already emits a discrete "uploading" progress event before this runs,
    so the bar itself is not needed.

    Yields:
        None: Control returns to the caller while ``tqdm`` is replaced with
            a silent passthrough iterator.
    """

    try:
        from cfa.cloudops import blob as cloudops_blob
    except Exception:
        yield
        return

    original_tqdm = cloudops_blob.tqdm

    def silent(iterable: Any, *args: Any, **kwargs: Any) -> Iterator[Any]:
        return iter(iterable)

    cloudops_blob.tqdm = silent
    try:
        yield
    finally:
        cloudops_blob.tqdm = original_tqdm
