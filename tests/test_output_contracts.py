from __future__ import annotations

from pathlib import Path

import pytest
from mrp.runtime import RunResult

from calibrationtools.mrp_runner import MRPOutputRunner
from calibrationtools.output_contracts import CSVTableOutputContract


def test_csv_table_output_contract_reads_columns(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "report.csv").write_text(
        "t_lower,t_upper,count\n0,1,10\n1,2,11\n",
        encoding="utf-8",
    )

    contract = CSVTableOutputContract(
        filename="report.csv",
        output_name="aggregated_deaths_report",
    )

    assert contract.read_output_dir(output_dir) == {
        "aggregated_deaths_report": {
            "t_lower": ["0", "1"],
            "t_upper": ["1", "2"],
            "count": ["10", "11"],
        }
    }


def test_csv_table_output_contract_raises_on_missing_file(tmp_path: Path):
    contract = CSVTableOutputContract(
        filename="report.csv",
        output_name="report",
    )

    with pytest.raises(FileNotFoundError, match="report.csv"):
        contract.read_output_dir(tmp_path)


def test_csv_table_output_contract_rejects_empty_csv():
    contract = CSVTableOutputContract(
        filename="report.csv",
        output_name="report",
    )

    with pytest.raises(ValueError, match="header"):
        contract.read_stdout("")


def test_csv_table_output_contract_strips_leading_logs_from_stdout():
    contract = CSVTableOutputContract(
        filename="report.csv",
        output_name="report",
        header_fields=("t_lower", "t_upper", "count"),
    )

    assert contract.read_stdout(
        "starting model\nwriting report\nt_lower,t_upper,count\n0,1,10\n"
    ) == {"report": {"t_lower": ["0"], "t_upper": ["1"], "count": ["10"]}}


def test_generic_mrp_runner_reads_table_from_stdout():
    def fake_mrp_run(config_path, overrides):
        return RunResult(
            exit_code=0,
            stdout=b"t_lower,t_upper,count\n0,1,10\n",
            stderr=b"",
        )

    runner = MRPOutputRunner(
        "model.mrp.toml",
        output_contract=CSVTableOutputContract(
            filename="report.csv",
            output_name="report",
        ),
        mrp_run_func=fake_mrp_run,
    )

    assert runner.simulate({"seed": 1}) == {
        "report": {"t_lower": ["0"], "t_upper": ["1"], "count": ["10"]}
    }


def test_generic_mrp_runner_reads_table_from_output_dir(tmp_path: Path):
    output_dir = tmp_path / "output"

    def fake_mrp_run(config_path, overrides, output_dir=None):
        assert output_dir is not None
        path = Path(output_dir)
        path.mkdir(parents=True)
        (path / "report.csv").write_text(
            "t_lower,t_upper,count\n0,1,10\n",
            encoding="utf-8",
        )
        return RunResult(exit_code=0, stdout=b"", stderr=b"")

    runner = MRPOutputRunner(
        "model.mrp.toml",
        output_contract=CSVTableOutputContract(
            filename="report.csv",
            output_name="report",
        ),
        mrp_run_func=fake_mrp_run,
    )

    assert runner.simulate({"seed": 1}, output_dir=output_dir) == {
        "report": {"t_lower": ["0"], "t_upper": ["1"], "count": ["10"]}
    }
