import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from calibrationtools import Particle
from calibrationtools.particle_runner import (
    ParticleRunner,
    _run_particles_helper,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def particle() -> Particle:
    return Particle({"x": 42})


@pytest.fixture
def multi_param_particle() -> Particle:
    return Particle({"rate": 0.5, "seed": 99})


def identity_params(particle: Particle) -> dict:
    """particles_to_params that returns the particle's own state."""
    return dict(particle.items())


# ---------------------------------------------------------------------------
# ParticleRunner.__init__ / _set_worker_count
# ---------------------------------------------------------------------------


class TestParticleRunnerInit:
    def test_parallel_is_default(self):
        model = MagicMock()
        runner = ParticleRunner(
            model=model, particles_to_params=identity_params
        )
        assert runner.parallel is True

    def test_visible_progress_is_default(self):
        model = MagicMock()
        runner = ParticleRunner(
            model=model, particles_to_params=identity_params
        )
        assert runner.verbose is True

    def test_quiet_runner_draws_no_progress(self, capsys):
        """Let a caller that owns the terminal silence this runner's bar.

        A concurrent study renders one dashboard for every scenario, so a
        per-scenario bar drawn here overwrites it mid-frame.
        """

        model = MagicMock()
        runner = ParticleRunner(
            model=model,
            particles_to_params=identity_params,
            execution="serial",
            verbose=False,
        )

        runner.run([Particle({"p": 0.1})])

        assert capsys.readouterr().out == ""

    def test_serial_execution(self):
        model = MagicMock()
        runner = ParticleRunner(
            model=model,
            particles_to_params=identity_params,
            execution="serial",
        )
        assert runner.parallel is False

    def test_explicit_max_workers(self):
        model = MagicMock()
        runner = ParticleRunner(
            model=model, particles_to_params=identity_params, max_workers=4
        )
        assert runner.max_workers == 4

    def test_zero_max_workers_uses_cpu_count(self):
        model = MagicMock()
        with patch(
            "calibrationtools.particle_runner.os.cpu_count",
            return_value=8,
        ):
            runner = ParticleRunner(
                model=model, particles_to_params=identity_params, max_workers=0
            )
        assert runner.max_workers == 8

    def test_none_max_workers_uses_cpu_count(self):
        model = MagicMock()
        with patch(
            "calibrationtools.particle_runner.os.cpu_count",
            return_value=4,
        ):
            runner = ParticleRunner(
                model=model,
                particles_to_params=identity_params,
                max_workers=None,
            )
        assert runner.max_workers == 4

    def test_fallback_to_os_cpu_count(self):
        model = MagicMock()
        with patch(
            "calibrationtools.particle_runner.os.cpu_count", return_value=2
        ):
            runner = ParticleRunner(
                model=model,
                particles_to_params=identity_params,
                max_workers=None,
            )
        assert runner.max_workers == 2

    def test_fallback_to_one_when_cpu_count_unavailable(self):
        model = MagicMock()
        with patch(
            "calibrationtools.particle_runner.os.cpu_count",
            return_value=None,
        ):
            runner = ParticleRunner(
                model=model,
                particles_to_params=identity_params,
                max_workers=None,
            )
        assert runner.max_workers == 1


# ---------------------------------------------------------------------------
# ParticleRunner.run_particle
# ---------------------------------------------------------------------------


class TestRunParticle:
    def test_calls_particles_to_params_and_simulate(self, particle):
        model = MagicMock()
        p2p = MagicMock(return_value={"x": 42})
        runner = ParticleRunner(model=model, particles_to_params=p2p)
        particle = particle

        runner.run_particle(particle)

        p2p.assert_called_once_with(particle=particle)
        model.simulate.assert_called_once_with({"x": 42})

    def test_kwargs_forwarded_to_particles_to_params(self, particle):
        model = MagicMock()
        p2p = MagicMock(return_value={})
        runner = ParticleRunner(model=model, particles_to_params=p2p)
        particle = particle

        runner.run_particle(particle, extra="kwarg")

        p2p.assert_called_once_with(particle=particle, extra="kwarg")

    def test_simulate_receives_correct_params(self, multi_param_particle):
        model = MagicMock()
        params = {"rate": 0.5, "seed": 99}
        runner = ParticleRunner(
            model=model, particles_to_params=lambda particle: params
        )
        runner.run_particle(multi_param_particle)
        model.simulate.assert_called_once_with(params)


# ---------------------------------------------------------------------------
# ParticleRunner._evaluate_particle_chunk
# ---------------------------------------------------------------------------


class TestEvaluateParticleChunk:
    def test_runs_all_particles_in_chunk(self, particle):
        model = MagicMock()
        runner = ParticleRunner(
            model=model, particles_to_params=identity_params
        )
        particles = [particle for i in range(3)]

        runner._evaluate_particle_chunk(particles)

        assert model.simulate.call_count == 3

    def test_empty_chunk_makes_no_calls(self):
        model = MagicMock()
        runner = ParticleRunner(
            model=model, particles_to_params=identity_params
        )

        runner._evaluate_particle_chunk([])

        model.simulate.assert_not_called()

    def test_returns_list_of_results(self, particle):
        model = MagicMock()
        model.simulate.return_value = "result"
        runner = ParticleRunner(
            model=model, particles_to_params=identity_params
        )
        particles = [particle for i in range(2)]

        result = runner._evaluate_particle_chunk(particles)

        assert isinstance(result, list)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# ParticleRunner.run – serial path
# ---------------------------------------------------------------------------


class TestRunSerial:
    def test_all_particles_simulated(self, particle):
        model = MagicMock()
        runner = ParticleRunner(
            model=model,
            particles_to_params=identity_params,
            execution="serial",
        )
        particles = [particle for i in range(5)]

        runner.run(particles)

        assert model.simulate.call_count == 5

    def test_empty_particle_list(self):
        model = MagicMock()
        runner = ParticleRunner(
            model=model,
            particles_to_params=identity_params,
            execution="serial",
        )
        runner.run([])  # should not raise
        model.simulate.assert_not_called()


# ---------------------------------------------------------------------------
# ParticleRunner.run – parallel path
# ---------------------------------------------------------------------------


class TestRunParallel:
    def test_all_particles_simulated(self, particle):
        model = MagicMock()
        runner = ParticleRunner(
            model=model,
            particles_to_params=identity_params,
            execution="parallel",
            max_workers=2,
        )
        particles = [particle for i in range(4)]

        runner.run(particles)

        assert model.simulate.call_count == 4

    def test_empty_particle_list(self):
        model = MagicMock()
        runner = ParticleRunner(
            model=model,
            particles_to_params=identity_params,
            execution="parallel",
        )
        runner.run([])  # should not raise
        model.simulate.assert_not_called()


# ---------------------------------------------------------------------------
# run_particles (async helper)
# ---------------------------------------------------------------------------


class TestRunParticlesAsync:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_worker_called_once_per_particle_with_chunksize_1(self, particle):
        worker = MagicMock(return_value=None)
        particles = [particle for i in range(3)]
        executor = ThreadPoolExecutor(max_workers=2)

        self._run(
            _run_particles_helper(
                particles=particles,
                executor=executor,
                worker=worker,
                chunksize=1,
            )
        )

        assert worker.call_count == 3

    def test_worker_called_once_per_chunk_with_larger_chunksize(
        self, particle
    ):
        worker = MagicMock(return_value=None)
        particles = [particle for i in range(6)]
        executor = ThreadPoolExecutor(max_workers=2)

        self._run(
            _run_particles_helper(
                particles=particles,
                executor=executor,
                worker=worker,
                chunksize=3,
            )
        )

        assert worker.call_count == 2

    def test_empty_particle_list(self):
        worker = MagicMock(return_value=None)
        executor = ThreadPoolExecutor(max_workers=1)

        self._run(
            _run_particles_helper(
                particles=[],
                executor=executor,
                worker=worker,
                chunksize=1,
            )
        )

        worker.assert_not_called()

    def test_default_reporter_created_when_none_provided(self, particle):
        worker = MagicMock(return_value=None)
        particles = [particle]
        executor = ThreadPoolExecutor(max_workers=1)

        # Should not raise even with reporter=None
        self._run(
            _run_particles_helper(
                particles=particles,
                executor=executor,
                worker=worker,
                reporter=None,
            )
        )

        worker.assert_called_once()
