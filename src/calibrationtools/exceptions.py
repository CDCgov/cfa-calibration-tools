class SimulationCancelledError(RuntimeError):
    """Raised when an in-flight simulation is cancelled on purpose."""

    def __init__(self, run_id: str | None = None):
        self.run_id = run_id
        message = (
            f"Simulation cancelled for run {run_id}."
            if run_id is not None
            else "Simulation cancelled."
        )
        super().__init__(message)


class CloudRunnerStateError(RuntimeError):
    """Raised when the cloud runner cannot honor a state-management call.

    Used (for example) when an admission-control slot cannot be released
    because the controller is in an unexpected state. Surfacing a typed
    error lets the sampler distinguish operational shutdown from real
    capacity-accounting bugs.
    """

    def __init__(self, message: str, *, run_id: str | None = None) -> None:
        self.run_id = run_id
        super().__init__(message)
