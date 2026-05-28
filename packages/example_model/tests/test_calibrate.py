import pytest
from example_model import calibrate


def test_calibrate_parser_accepts_slot_lookahead():
    args = calibrate.build_parser().parse_args(["--slot-lookahead", "3"])

    assert args.slot_lookahead == 3


def test_calibrate_parser_rejects_non_positive_slot_lookahead():
    with pytest.raises(SystemExit):
        calibrate.build_parser().parse_args(["--slot-lookahead", "0"])


def test_calibrate_main_passes_slot_lookahead(monkeypatch):
    captured: dict[str, int | None] = {}

    def fake_run_calibration(slot_lookahead=None):
        captured["slot_lookahead"] = slot_lookahead

    monkeypatch.setattr(calibrate, "run_calibration", fake_run_calibration)

    calibrate.main(["--slot-lookahead", "4"])

    assert captured == {"slot_lookahead": 4}
