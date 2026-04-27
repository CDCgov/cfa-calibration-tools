from pathlib import Path

from example_model.cloud_auto_size import _run_probe_simulation, main


def test_main_uses_shared_child_runner(monkeypatch):
    captured = {}

    def fake_child_main(run_probe_simulation):
        captured["run_probe_simulation"] = run_probe_simulation

    monkeypatch.setattr(
        "example_model.cloud_auto_size.run_memory_probe_child_main",
        fake_child_main,
    )

    main()

    assert captured["run_probe_simulation"] is _run_probe_simulation


def test_run_probe_simulation_writes_example_output(tmp_path):
    _run_probe_simulation(
        {
            "seed": 123,
            "max_gen": 3,
            "n": 3,
            "p": 0.5,
            "max_infect": 500,
        },
        "probe-1",
        tmp_path,
    )

    output_path = Path(tmp_path) / "output.csv"
    assert output_path.read_text().startswith("generation,population\n")
