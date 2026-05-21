import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Concatenate, Literal

from .async_runner import run_coroutine_from_sync
from .particle import Particle
from .sampler_reporting import SamplerReporter


async def _run_particles_helper(
    particles: list[Particle],
    executor: ThreadPoolExecutor,
    worker: Callable[[list[Particle]], Any],
    chunksize: int = 1,
    reporter: SamplerReporter | None = None,
):
    """
    Async function to call a worker function on each member of a list of Particles in parallel

    Args:
        particles (list[Particle]): List of particles to run the worker function on
        executor (ThreadPoolExecutor): Executor to run the worker function on
        worker (Callable[[list[Particle]], Any]): Worker function to run on each particle
        chunksize (int): Number of particles to run in each chunk. Defaults to 1.
        reporter (SamplerReporter | None): Optional reporter to use for progress reporting. Defaults to None, in which case a new SamplerReporter will be created.

    """
    if reporter is None:
        reporter = SamplerReporter(verbose=True)

    with reporter.create_task_progress() as progress:
        assert executor is not None
        start = time.time()
        handle = reporter.start_task(
            description="Simulating results from model... ",
            progress=progress,
            total=len(particles),
        )
        loop = asyncio.get_running_loop()
        particle_chunks = [
            particles[index : index + chunksize]
            for index in range(0, len(particles), chunksize)
        ]
        tasks = []
        for chunk in particle_chunks:
            task = loop.run_in_executor(executor, worker, chunk)
            tasks.append((task, chunk))

        try:
            for task, chunk in tasks:
                _chunk_results = await task
                reporter.advance(handle)
        finally:
            for task, _ in tasks:
                task.cancel()

        end = time.time()
        reporter.print_run_summary(end - start, process_name="Simulations")


class ParticleRunner:
    """
    ParticleRunner is a utility class for running Particle objects through a user-supplied model. It supports both parallel and serial execution modes, and can be configured with a custom function to convert Particle instances into the parameter dictionaries expected by the model.
    This class is designed to keep the details of parallel execution and model interfacing
    contained in one place, so that the ABCSampler can focus on the logic of the sampling algorithm itself.
    Args:
        model (Any): User-supplied model object that has a .simulate(params: dict) method for running simulations.
        particles_to_params (Callable[Concatenate[Particle, ...], dict]): Function that converts a Particle instance into a parameter dictionary for the model. It should accept the particle as its first argument and return a dict of parameters to be passed to the model's simulate method.
        execution (Literal["parallel", "serial"], optional): Whether to run particles in parallel using threads or serially. Defaults to "parallel".
        max_workers (int | None, optional): Maximum number of worker threads to use for parallel execution. If None or 0, it will be set to the number of CPU cores available. Defaults to None.
    """

    def __init__(
        self,
        model: Any,
        particles_to_params: Callable[Concatenate[Particle, ...], dict],
        execution: Literal["parallel", "serial"] = "parallel",
        max_workers: int | None = None,
    ):
        self.model = model
        self.parallel = execution == "parallel"
        self._set_worker_count(max_workers)
        self.particles_to_params = particles_to_params

    def _set_worker_count(self, max_workers: int | None = None):
        """
        Set the number of workers to use for parallel execution. If max_workers is None or 0, it will be set to the number of CPU cores available.
        Args:
            max_workers (int | None): The maximum number of workers to use for parallel execution. If None or 0, it will be set to the number of CPU cores available.
        """
        if not max_workers:
            self.max_workers = os.cpu_count() or 1
        else:
            self.max_workers = max_workers

    def run_particle(self, particle: Particle, **kwargs):
        """
        Run a single particle through the model.
        Args:
            particle (Particle): Particle to run through the model
            **kwargs: Additional keyword arguments forwarded into particle evaluation
        """
        particle_params = self.particles_to_params(particle=particle, **kwargs)
        self.model.simulate(particle_params)

    def _evaluate_particle_chunk(
        self, particles: list[Particle], **kwargs
    ) -> list[Any]:
        """Evaluate a chunk of proposed particles serially.

        This helper keeps chunk evaluation reusable between the serial and
        threaded batch-processing paths.

        Args:
            particles (list[Particle]): Proposed particles to score.
            **kwargs: Additional keyword arguments forwarded into particle evaluation.
        Returns:
            list[Any]: List of model outputs corresponding to the proposed particles.
        """

        return [
            self.run_particle(particle=particle, **kwargs)
            for particle in particles
        ]

    def run(self, particles: list[Particle]):
        """
        Call the particle evaluation for each member of a list of Particles
        Args:
            particles (list[Particle]): List of particles to run the worker function on
        """
        if self.parallel:
            run_coroutine_from_sync(
                lambda: _run_particles_helper(
                    executor=ThreadPoolExecutor(max_workers=self.max_workers),
                    worker=self._evaluate_particle_chunk,
                    chunksize=1,
                    reporter=SamplerReporter(verbose=True),
                    particles=particles,
                )
            )
        else:
            reporter = SamplerReporter(verbose=True)
            with reporter.create_task_progress() as progress:
                start = time.time()
                handle = reporter.start_task(
                    description="Simulating results from model... ",
                    progress=progress,
                    total=len(particles),
                )
                for particle in particles:
                    self.run_particle(particle=particle)
                    reporter.advance(handle)

                end = time.time()
                reporter.print_run_summary(
                    end - start, process_name="Simulations"
                )
