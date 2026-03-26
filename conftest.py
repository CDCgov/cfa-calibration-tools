from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
for rel_path in ("src", "packages/example_model/src"):
    path = str(ROOT / rel_path)
    if path not in sys.path:
        sys.path.insert(0, path)
