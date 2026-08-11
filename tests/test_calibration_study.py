import pytest

from calibrationtools import CalibrationScenario, CalibrationStudy
from calibrationtools.cloud_executor import CloudExecutor


class CloneableExecutor(CloudExecutor):
    async def execute_tasks(self, tasks, *, progress_callback=None):
        raise AssertionError("study validation must not execute cloud work")

    def clone_for_scenario(self, scenario_name: str):
        return self


def test_calibration_scenario_copies_parameter_payload() -> None:
    parameters = {"beta": 0.2}

    scenario = CalibrationScenario("baseline", parameters)
    parameters["beta"] = 0.4

    assert scenario.parameters == {"beta": 0.2}


@pytest.mark.parametrize(
    ("scenarios", "max_concurrent_scenarios", "message"),
    [
        ([], 1, "at least one"),
        (
            [CalibrationScenario("same"), CalibrationScenario("same")],
            1,
            "unique",
        ),
        ([CalibrationScenario("only")], 0, "positive"),
    ],
)
def test_calibration_study_validates_scenarios_and_concurrency(
    scenarios, max_concurrent_scenarios, message
) -> None:
    with pytest.raises(ValueError, match=message):
        CalibrationStudy(
            scenarios=scenarios,
            sampler_factory=lambda _: None,
            cloud_executor=CloneableExecutor(),
            max_concurrent_scenarios=max_concurrent_scenarios,
        )
