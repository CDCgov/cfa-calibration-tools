import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def dumps_json(value: Any, *, indent: int = 2) -> str:
    return json.dumps(to_jsonable(value), indent=indent, sort_keys=True)
