import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable

from .particle import Particle

class LocalParallelExecutor():
    """Local parallel executor using multiprocessing.

    This executor uses the local machine's CPU cores to process particles
    in parallel using Python's multiprocessing module.
    """

    def __init__(
        self,
        max_workers: int | None = None,
        chunk_size: int = 1,
    ) -> None:
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        if sys.platform.startswith("linux"):
            import multiprocessing

            multiprocessing.set_start_method("spawn", force=True)

    async def batch_submit_particles(
        self,
        particles: list[Particle],
        particle_to_float: Callable[[Particle], float],
        **kwargs: Any,
    ) -> list[float]:
        """Execute particle updates using local multiprocessing."""

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            total_tasks = len(particles)
            actual_workers = (
                min(self.max_workers, (max(mp.cpu_count(), 1)))
                if self.max_workers
                else (mp.cpu_count() or 1)
            )

            return executor.map(
                particle_to_float, particles, chunksize=self.chunk_size
            )
