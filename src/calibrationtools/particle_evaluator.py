"""Evaluate particles by running the model and scoring its outputs."""

import asyncio
import inspect
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from mrp import MRPModel

from .async_runner import run_coroutine_from_sync
from .json_utils import dumps_json, to_jsonable
from .particle import Particle

EVALUATION_CONTEXT_ARG = "evaluation_context"
# Deprecated compatibility alias for historical internal callers.
EVALUATION_CONTEXT_KEY = "_calibrationtools_evaluation_context"


def build_evaluation_context_kwargs(
    *,
    generation: int,
    proposal_index: int,
    attempt_index: int = 0,
    base_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return kwargs with an evaluation context attached under the public key.

    Runners call this before invoking ``particle_to_distance(...)`` so the
    sampler-side context and ``ParticleEvaluator`` stay in sync; both the
    batched and particlewise runners construct identical dicts and this
    keeps that contract in one place.
    """
    kwargs = dict(base_kwargs) if base_kwargs else {}
    kwargs[EVALUATION_CONTEXT_ARG] = {
        "generation_index": generation,
        "proposal_index": proposal_index,
        "attempt_index": attempt_index,
    }
    return kwargs


class ParticleEvaluator:
    """Evaluate particles by running the model and scoring its outputs."""

    def __init__(
        self,
        particles_to_params: Callable[..., dict],
        outputs_to_distance: Callable[..., float],
        target_data: Any,
        model_runner: MRPModel,
        artifacts_dir: Path | str | None = None,
    ) -> None:
        self.particles_to_params = particles_to_params
        self.outputs_to_distance = outputs_to_distance
        self.target_data = target_data
        self.model_runner = model_runner
        self.artifacts_dir = (
            Path(artifacts_dir) if artifacts_dir is not None else None
        )

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    @staticmethod
    def _build_run_id(context: dict[str, int] | None) -> str | None:
        if context is None:
            return None
        base = (
            f"gen-{context['generation_index'] + 1}_"
            f"particle-{context['proposal_index'] + 1}"
        )
        if context.get("attempt_index", 0) <= 0:
            return base
        return f"{base}-attempt-{context['attempt_index'] + 1}"

    @staticmethod
    def _build_direct_run_id() -> str:
        return f"direct-{uuid4().hex}"

    @staticmethod
    def _resolve_evaluation_context(
        *,
        evaluation_context: dict[str, int] | None,
        kwargs: dict[str, Any],
    ) -> dict[str, int] | None:
        legacy_context = kwargs.pop(EVALUATION_CONTEXT_KEY, None)
        if evaluation_context is not None and legacy_context is not None:
            raise TypeError(
                f"Pass either evaluation_context or {EVALUATION_CONTEXT_KEY}, not both."
            )
        if evaluation_context is not None:
            return evaluation_context
        return legacy_context

    @staticmethod
    def _build_simulate_kwargs(
        simulate: Callable[..., Any],
        *,
        input_path: Path | None,
        output_dir: Path | None,
        run_id: str | None,
    ) -> dict[str, Any]:
        signature = inspect.signature(simulate)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        simulate_kwargs: dict[str, Any] = {}
        if input_path is not None and (
            accepts_kwargs or "input_path" in signature.parameters
        ):
            simulate_kwargs["input_path"] = input_path
        if output_dir is not None and (
            accepts_kwargs or "output_dir" in signature.parameters
        ):
            simulate_kwargs["output_dir"] = output_dir
        if run_id is not None and (
            accepts_kwargs or "run_id" in signature.parameters
        ):
            simulate_kwargs["run_id"] = run_id
        return simulate_kwargs

    def _stage_simulation_io(
        self,
        params: dict[str, Any],
        *,
        context: dict[str, int] | None,
    ) -> tuple[dict[str, Any], Path | None, Path | None, str | None]:
        run_id = self._build_run_id(context)
        staged_params = to_jsonable(params)

        if self.artifacts_dir is None:
            if run_id is None:
                # No artifacts to lay out and no context to build a run_id
                # from; caller will get run_id=None and handle it.
                return staged_params, None, None, None
            return staged_params, None, None, run_id

        if context is None:
            # Direct callers are allowed to persist artifacts without
            # constructing sampler-specific generation metadata. Route
            # those runs into a separate namespace so they do not collide
            # with generation-indexed sampler artifacts.
            generation_name = "direct"
            run_id = self._build_direct_run_id()
        else:
            generation_index = context["generation_index"]
            generation_name = f"generation-{generation_index + 1}"
            assert (
                run_id is not None
            )  # narrow for type checker; context is set

        staged_params["run_id"] = run_id
        input_dir = self.artifacts_dir / "input" / generation_name
        output_dir = self.artifacts_dir / "output" / generation_name / run_id
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        input_path = input_dir / f"{run_id}.json"
        input_path.write_text(dumps_json(staged_params) + "\n")
        return staged_params, input_path, output_dir, run_id

    @staticmethod
    async def _await_result(result: Any) -> Any:
        return await result

    def _allows_sync_bridge(self) -> bool:
        return bool(
            getattr(
                self.model_runner,
                "allow_simulate_from_sync_bridge",
                False,
            )
        )

    def _simulate(
        self,
        params: dict[str, Any],
        *,
        input_path: Path | None,
        output_dir: Path | None,
        run_id: str | None,
    ) -> Any:
        simulate_async = getattr(self.model_runner, "simulate_async", None)
        simulate_from_sync = getattr(
            self.model_runner, "simulate_from_sync", None
        )
        if bool(
            getattr(self.model_runner, "prefer_simulate_async", False)
        ) and callable(simulate_async):
            if callable(simulate_from_sync) and self._allows_sync_bridge():
                return simulate_from_sync(
                    params,
                    **self._build_simulate_kwargs(
                        simulate_from_sync,
                        input_path=input_path,
                        output_dir=output_dir,
                        run_id=run_id,
                    ),
                )
            return run_coroutine_from_sync(
                lambda: self._simulate_async(
                    simulate_async,
                    params,
                    input_path=input_path,
                    output_dir=output_dir,
                    run_id=run_id,
                )
            )

        simulate = getattr(self.model_runner, "simulate", None)
        if callable(simulate):
            result = simulate(
                params,
                **self._build_simulate_kwargs(
                    simulate,
                    input_path=input_path,
                    output_dir=output_dir,
                    run_id=run_id,
                ),
            )
            if inspect.isawaitable(result):
                return run_coroutine_from_sync(
                    lambda: self._await_result(result)
                )
            return result

        if callable(simulate_async):
            return run_coroutine_from_sync(
                lambda: self._simulate_async(
                    simulate_async,
                    params,
                    input_path=input_path,
                    output_dir=output_dir,
                    run_id=run_id,
                )
            )

        raise AttributeError(
            "model_runner must define simulate() or simulate_async()"
        )

    async def _simulate_async_preferred(
        self,
        params: dict[str, Any],
        *,
        input_path: Path | None,
        output_dir: Path | None,
        run_id: str | None,
    ) -> Any:
        simulate_async = getattr(self.model_runner, "simulate_async", None)
        if bool(
            getattr(self.model_runner, "prefer_simulate_async", False)
        ) and callable(simulate_async):
            return await self._simulate_async(
                simulate_async,
                params,
                input_path=input_path,
                output_dir=output_dir,
                run_id=run_id,
            )

        simulate = getattr(self.model_runner, "simulate", None)
        if callable(simulate):
            result = await asyncio.to_thread(
                lambda: simulate(
                    params,
                    **self._build_simulate_kwargs(
                        simulate,
                        input_path=input_path,
                        output_dir=output_dir,
                        run_id=run_id,
                    ),
                )
            )
            if inspect.isawaitable(result):
                return await result
            return result

        if callable(simulate_async):
            return await self._simulate_async(
                simulate_async,
                params,
                input_path=input_path,
                output_dir=output_dir,
                run_id=run_id,
            )

        raise AttributeError(
            "model_runner must define simulate() or simulate_async()"
        )

    async def _simulate_async(
        self,
        simulate_async: Callable[..., Any],
        params: dict[str, Any],
        *,
        input_path: Path | None,
        output_dir: Path | None,
        run_id: str | None,
    ) -> Any:
        result = simulate_async(
            params,
            **self._build_simulate_kwargs(
                simulate_async,
                input_path=input_path,
                output_dir=output_dir,
                run_id=run_id,
            ),
        )
        if inspect.isawaitable(result):
            return await result
        return result

    def simulate(
        self,
        particle: Particle,
        *,
        evaluation_context: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> Any:
        context = self._resolve_evaluation_context(
            evaluation_context=evaluation_context,
            kwargs=kwargs,
        )
        params = self.particles_to_params(particle, **kwargs)
        staged_params, input_path, output_dir, run_id = (
            self._stage_simulation_io(params, context=context)
        )
        outputs = self._simulate(
            staged_params,
            input_path=input_path,
            output_dir=output_dir,
            run_id=run_id,
        )
        if output_dir is not None:
            (output_dir / "result.json").write_text(dumps_json(outputs) + "\n")
        return outputs

    async def simulate_async(
        self,
        particle: Particle,
        *,
        evaluation_context: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> Any:
        context = self._resolve_evaluation_context(
            evaluation_context=evaluation_context,
            kwargs=kwargs,
        )
        params = self.particles_to_params(particle, **kwargs)
        staged_params, input_path, output_dir, run_id = (
            self._stage_simulation_io(params, context=context)
        )
        outputs = await self._simulate_async_preferred(
            staged_params,
            input_path=input_path,
            output_dir=output_dir,
            run_id=run_id,
        )
        if output_dir is not None:
            await asyncio.to_thread(
                lambda: (output_dir / "result.json").write_text(
                    dumps_json(outputs) + "\n"
                )
            )
        return outputs

    def distance(
        self,
        particle: Particle,
        *,
        evaluation_context: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> float:
        outputs = self.simulate(
            particle,
            evaluation_context=evaluation_context,
            **kwargs,
        )
        return self.outputs_to_distance(outputs, self.target_data)

    async def distance_async(
        self,
        particle: Particle,
        *,
        evaluation_context: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> float:
        outputs = await self.simulate_async(
            particle,
            evaluation_context=evaluation_context,
            **kwargs,
        )
        return self.outputs_to_distance(outputs, self.target_data)
