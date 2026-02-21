# Example model

This python package provides a simple branching process model intended to be used with the `mrp` (model runner protocol) package.

This README will describe multiple ways to run the model.

First open the Python interactive shell within the `uv` environment:

```bash
uv sync --all-packages
uv run python
```
The following two lines of code are sufficient to run the example model using the static method (i.e., calling `Binom_BP_Model.simulate()`).
```python
from example_model import Binom_BP_Model
Binom_BP_Model.simulate({"seed": 123, "max_gen": 15, "n": 3, "p": 0.5, "max_infect": 500})
```
This should yield the following output: `[1, 2, 1, 1, 1, 2, 4, 8, 15, 24, 28, 38, 58, 86, 126]`

An equivalent approach is to read the model input from `defaults.json`. (The code here assumes working from the root of the repo.)

```python
import json
from example_model import Binom_BP_Model
model_inputs = json.load(open("./packages/example_model/defaults.json"))
Binom_BP_Model.simulate(model_inputs)
```
To use the MRP functionality, create an environment that specifies the inputs and use the `.run()` method:
```python
from mrp import Environment
from example_model import Binom_BP_Model
model_inputs = {"max_gen": 15, "n": 3, "p": 0.5, "max_infect": 500}
env = Environment({"input": model_inputs})
Binom_BP_Model(env).run()
```
The above examples are very similar to those included in `scripts/direct_runner.py`, which can be run (from the root of the repo) with the following:
```bash
uv sync --all-packages
uv run python scripts/direct_runner.py
```
Additionally, as described in the repo-level README, the model can be run as specified in the `example_model.mrp.toml`, which can be run as follows:
```bash
uv sync --all-packages
uv run mrp run example_model.mrp.toml
```
