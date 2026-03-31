from pathlib import Path

_WORKSPACE_PACKAGE = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "example_model"
    / "src"
    / "example_model"
)
if _WORKSPACE_PACKAGE.is_dir():
    __path__.append(str(_WORKSPACE_PACKAGE))

from .example_model import Binom_BP_Model  # noqa: E402

__all__ = ["Binom_BP_Model"]
