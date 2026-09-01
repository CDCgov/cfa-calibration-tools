"""Azure Batch implementation of the cloud acceptance-task contract.

The implementation is split by concern so each file stays reviewable:

- ``_console``: quiets/redraws ``cfa-cloudops`` console and ``tqdm`` output.
- ``_node_health``: polls and interprets Batch pool/node state.
- ``_image``: resolves the container registry and publishes worker images.
- ``_job``: uploads task chunks, submits/polls/downloads/cleans up a job.
- ``_executor``: ``AzureBatchExecutor``, the public class that orchestrates
  the modules above behind the ``CloudExecutor`` contract.
"""

from ._executor import AzureBatchExecutor

__all__ = ["AzureBatchExecutor"]
