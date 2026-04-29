import csv
import json

from example_model.direct_runner import ExampleModelDirectRunner
from example_model.example_model import Binom_BP_Model

from calibrationtools.particle import Particle
from calibrationtools.particle_evaluator import ParticleEvaluator
from calibrationtools.particle_reader import ParticleReader


def test_direct_runner_loads_staged_input_and_writes_output_csv(tmp_path):
    staged_input = {
        "seed": 123,
        "max_gen": 4,
        "n": 3,
        "p": 0.5,
        "max_infect": 500,
        "run_id": "gen_0_particle_0_attempt_0",
    }
    input_path = tmp_path / "input.json"
    output_dir = tmp_path / "output"
    input_path.write_text(json.dumps(staged_input))

    runner = ExampleModelDirectRunner()

    result = runner.simulate(
        {
            "seed": 999,
            "max_gen": 1,
            "n": 0,
            "p": 0.0,
            "max_infect": 1,
        },
        input_path=input_path,
        output_dir=output_dir,
        run_id="gen_0_particle_0_attempt_0",
    )

    assert result == Binom_BP_Model.simulate(staged_input)
    with (output_dir / "output.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == [
        {"generation": str(index), "population": str(population)}
        for index, population in enumerate(result)
    ]


def test_direct_runner_adds_run_id_without_mutating_params(tmp_path):
    params = {
        "seed": 123,
        "max_gen": 3,
        "n": 1,
        "p": 0.0,
        "max_infect": 500,
    }
    runner = ExampleModelDirectRunner()

    result = runner.simulate(
        params,
        output_dir=tmp_path,
        run_id="gen_0_particle_1_attempt_0",
    )

    assert result == Binom_BP_Model.simulate(
        {**params, "run_id": "gen_0_particle_1_attempt_0"}
    )
    assert "run_id" not in params
    assert (tmp_path / "output.csv").is_file()


def test_direct_runner_writes_sampler_artifact_tree(tmp_path):
    runner = ExampleModelDirectRunner()
    evaluator = ParticleEvaluator(
        particle_reader=ParticleReader(
            particle_param_names=["p"],
            default_params = {
                "seed": 123,
                "max_gen": 3,
                "n": 3,
                "p": 0.1,
                "max_infect": 500,
            },
        ),
        outputs_to_distance=lambda outputs, target: 0.0,
        target_data=None,
        model_runner=runner,
        artifacts_dir=tmp_path,
    )

    evaluator.distance(
        Particle({"p": 0.5}),
        evaluation_context={
            "generation_index": 0,
            "proposal_index": 0,
            "attempt_index": 0,
        },
    )

    run_id = "gen_0_particle_0_attempt_0"
    assert (tmp_path / "input" / "generation-0" / f"{run_id}.json").is_file()
    output_dir = tmp_path / "output" / "generation-0" / run_id
    assert (output_dir / "output.csv").is_file()
    assert (output_dir / "result.json").is_file()
